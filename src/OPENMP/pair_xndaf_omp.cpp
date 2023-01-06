// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Yuansheng Zhao
------------------------------------------------------------------------- */

#include "pair_xndaf_omp.h"

#include <cmath>

#include <cstring>
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "suffix.h"
#include "memory.h"
#include "domain.h"
#ifdef XNDAF_DEBUG
#include <chrono>
#include <iostream>
#endif
#ifdef XNDAF_OMP_DEBUG
#include <iostream>
#endif

#include "omp_compat.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairXNDAFOMP::PairXNDAFOMP(LAMMPS *lmp) : PairXNDAF(lmp),ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
}

void PairXNDAFOMP::compute(int eflag, int vflag)
{
  int i,j;
  // if(comm->me==0) utils::logmesg(lmp,"call compute\n");
  ev_init(eflag, vflag);
  // calc sq and generate force table;
  if(ncall%update_interval==0){
    compute_sq();
    generateForceTable();
  }
  if(comm->me==0 && ncall%output_interval==0){
    // utils::logmesg(lmp,"output sq and gr\n");
    FILE *fp=fopen(sqout,"w");
    for(i=0;i<nbin_q;i++)
    {
      fprintf(fp,"%lf %lf %lf %lf %lf",ssq[i][0],ssq[i][1],ssq[i][2],iiq[i][0],iiq[i][1]);
      for(j=0;j<npair;j++){
        fprintf(fp," %lf",ssq[i][3+j]);
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
    fp=fopen(grout,"w");
    for(i=0;i<nbin_r;i++)
    {
      fprintf(fp,"%lf",(i+.5)*ddr);
      for(j=0;j<npair;j++){
        fprintf(fp," %lf",ggr[i][j]);
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
  }

  ncall++;


  // const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    // ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (force->newton_pair) eval<1>(ifrom, ito, thr);
    else eval<0>(ifrom, ito, thr);

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
  eng_vdwl=localerg;
}

template <int NEWTON_PAIR>
void PairXNDAFOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int * _noalias const type = atom->type;
  const double * _noalias const special_lj = force->special_lj;
  const int * _noalias const ilist = list->ilist;
  const int * _noalias const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;

  double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
  double rsq,forcelj,factor_lj,fpair;

  const int nlocal = atom->nlocal;
  int j,jj,jnum,jtype;
  int tbi;

  #ifdef XNDAF_DEBUG
  auto t1=std::chrono::high_resolution_clock::now();
  #endif

  // loop over neighbors of my atoms

  const double cutsqall = force_cutoff_sq;// cutsq[0][0];
  for (int ii = iifrom; ii < iito; ++ii) {
    const int i = ilist[ii];
    const int itype = type[i];
    const int    * _noalias const jlist = firstneigh[i];

    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    jnum = numneigh[i];
    fxtmp=fytmp=fztmp=0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsqall) {
        rsq=sqrt(rsq);
        tbi=(int)(rsq/ddr);
        if(tbi>=nbin_r) continue;
        fpair = getForce(tbi,typ2pair[itype][jtype])*factor_lj;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= delx*fpair;
          f[j].y -= dely*fpair;
          f[j].z -= delz*fpair;
        }
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
  #ifdef XNDAF_DEBUG
  auto t2=std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "total compute loop " << time_span.count() << " s\n";
  #endif
  // if (vflag_fdotr) virial_fdotr_compute();

}

void PairXNDAFOMP::coeff(int narg, char **arg)
{
  PairXNDAF::coeff(narg,arg);
  memory->create(cnt_omp_private,comm->nthreads,nbin_r,npair,"rmdfomp:cnt");
}

void PairXNDAFOMP::compute_sq()
{
  #ifdef XNDAF_DEBUG
  auto t1=std::chrono::high_resolution_clock::now();
  #endif
  // if(comm->me==0) utils::logmesg(lmp,"call compute_sq\n");
  int src,i,j;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

  // zero the histogram counts
  for (i = 0; i < npair; i++)
    for (j = 0; j < nbin_r; j++)
      cnt[j][i] = 0;


#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
  #ifdef XNDAF_OMP_DEBUG
  std::cout << tid << " " << ifrom << " "<<ito <<"\n";
  #endif
    if (force->newton_pair) evalsq<1>(ifrom, ito, thr, cnt_omp_private[tid]);
    else evalsq<0>(ifrom, ito, thr, cnt_omp_private[tid]);

#if defined(_OPENMP)
#pragma omp critical 
#endif
    {
      for (i = 0; i < npair; i++)
        for (j = 0; j < nbin_r; j++)
        {
          cnt[j][i] += cnt_omp_private[tid][j][i];
        }
    }

  } // end of omp parallel region  

  // sum histograms across procs

  MPI_Allreduce(cnt[0],cnt_all[0],npair*nbin_r,MPI_INT,MPI_SUM,world);

  // convert counts to g(r) and coord(r) and copy into output array
  // vfrac = fraction of volume in shell m
  // npairs = number of pairs, corrected for duplicates
  // duplicates = pairs in which both atoms are the same
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for (int gk=0;gk<npair;gk++){
    for(int rk=0;rk<nbin_r;rk++){
      ggr[rk][gk]=cnt_all[rk][gk]*gnm[rk][gk];
      // array[rk][1+gk]=ggr[rk][gk];
    }
  }

  // gr -> sq
  const double vinv=natoms/((domain->xprd)*(domain->yprd)*(domain->zprd));
#if defined(_OPENMP)
#pragma omp parallel for private(src)
#endif
  for(int qk=0;qk<nbin_q;qk++){
    ssq[qk][1]=ssq[qk][2]=0;
    for (int gk=0;gk<npair;gk++){
      src=gk+3;
      ssq[qk][src]=0;
      for(int rk=0;rk<nbin_r;rk++){
        ssq[qk][src]+=(ggr[rk][gk]-vinv)*sinqr[qk][rk];
      }
      ssq[qk][1]+=ssq[qk][src]*sff[qk][gk];
      ssq[qk][2]+=ssq[qk][src]*sffn[gk];
    }
  }
  #ifdef XNDAF_DEBUG
  auto t2=std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "sq " << time_span.count() << " s\n";
  #endif
}

template <int NEWTON_PAIR>
void PairXNDAFOMP::evalsq(int iifrom, int iito, ThrData * const thr, int **counter)
{
  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const int * _noalias const type = atom->type;
  const int * _noalias const ilist = list->ilist;
  const int * _noalias const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;

  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq;

  const int nlocal = atom->nlocal;
  int j,jj,jnum,jtype;
  int ibin;

  for (int i = 0; i < npair; i++)
    for (j = 0; j < nbin_r; j++)
      counter[j][i] = 0;
      
  const double cutsqall = cutsq[0][0];
  for (int ii = iifrom; ii < iito; ++ii) {
    const int i = ilist[ii];
    const int itype = type[i];
    const int    * _noalias const jlist = firstneigh[i];

    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsqall) {
        rsq=sqrt(rsq);
        ibin=(int)(rsq/ddr);
        if (ibin < nbin_r) {
          if (NEWTON_PAIR || j < nlocal) counter[ibin][typ2pair[itype][jtype]]+=2;
          else counter[ibin][typ2pair[itype][jtype]]++;
        }
      }
    }
  }
}

void PairXNDAFOMP::generateForceTable()
{  
  #ifdef XNDAF_DEBUG
  auto t1=std::chrono::high_resolution_clock::now();
  #endif

  for(int i=0;i<npair;i++){
    for(int rk=0;rk<nbin_r;rk++){
      // frc[rk][i]=INFINITY;
#ifndef XNDAF_INSTANT_FORCE
      frc_allocated[rk][i]=0;
#else
      frc[rk][i]=0;
#endif
    }
  }
  // sq to iq and norm
  double nrsqrx=0,nrsqrn=0,normx=0,normn=0;
  // remove bias
  for(int i=0;i<nbin_q;i++){
    normx+=(iiq[i][0]=ssq[i][1]*kq[i][0]+mq[i][0])*wt[i][0];
    normn+=(iiq[i][1]=ssq[i][2]*kq[i][1]+mq[i][1])*wt[i][1];
  }
#if defined(_OPENMP)
#pragma omp parallel for
#endif
  for(int i=0;i<nbin_q;i++){
    iiq[i][0]-=normx ;
    iiq[i][1]-=normn;
  }
  //calc var
  for(int i=0;i<nbin_q;i++){
    nrsqrx+=iiq[i][0]*iiq[i][0]*wt[i][0];
    nrsqrn+=iiq[i][1]*iiq[i][1]*wt[i][1];
  }
  normx=sqrt(nrsqrx);
  normn=sqrt(nrsqrn);
  //inner prod
  double crsx=0,crsn=0;
  for(int i=0;i<nbin_q;i++){
    crsx+=iiq[i][0]*sqex[i][0]*wt[i][0];
    crsn+=iiq[i][1]*sqex[i][1]*wt[i][1];
  }
  crsx/=nrsqrx;
  crsn/=nrsqrn;  

  double diffx,diffn;
#if defined(_OPENMP)
#pragma omp parallel for private(diffx,diffn)
#endif
  for(int qk=0;qk<nbin_q;qk++){
    diffx=(sqex[qk][0]-iiq[qk][0]*crsx)*wk[qk][0]/normx;
    diffn=(sqex[qk][1]-iiq[qk][1]*crsn)*wk[qk][1]/normn;
    for(int i=0;i<npair;i++){
      force_qspace[i][qk]=diffx*sff_w[qk][i]+diffn*sffn_w[i];
    }
  }
  #ifdef XNDAF_INSTANT_FORCE
  #if defined(_OPENMP)
  #pragma omp parallel for
  #endif
  for(int i=0;i<npair;i++){
    for(int rk=0;rk<nbin_r;rk++){
      for(int qk=0;qk<nbin_q;qk++){
        frc[rk][i]+=force_qspace[i][qk]*dsicqr_dr_div_r[qk][rk];
      }
    }
  }
  #endif
  localerg=((1-crsx*normx)*factorx+(1-crsn*normn)*factorn)*atom->nlocal;

  #ifdef XNDAF_DEBUG
  auto t2=std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "force table " << time_span.count() << " s\n";
  #endif
}