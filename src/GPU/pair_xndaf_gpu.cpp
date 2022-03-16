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

#include "pair_xndaf_gpu.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "suffix.h"
#include "comm.h"

#include <cmath>

#ifdef XNDAF_DEBUG
#include <chrono>
#include <iostream>
#endif
#ifdef XNDAF_OMP_DEBUG
#include <iostream>
#endif

#include "omp_compat.h"
using namespace LAMMPS_NS;

int xndaf_gpu_init(const int ntypes, const int ntable, double host_cutsq,
           double host_dr, double *host_special_lj,
           const int inum, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           int &gpu_mode, FILE *screen);
void xndaf_gpu_clear();
void xndaf_gpu_sendTable(double **tb);
int ** xndaf_gpu_compute_n(const int ago, const int inum_full,
                           const int nall, double **host_x, int *host_type,
                           double *sublo, double *subhi, tagint *tag, int **nspecial,
                           tagint **special, const bool eflag, const bool vflag,
                           const bool eatom, const bool vatom, int &host_start,
                           int **ilist, int **jnum, const double cpu_time,
                           bool &success);
void xndaf_gpu_compute(const int ago, const int inum_full, const int nall,
                       double **host_x, int *host_type, int *ilist, int *numj,
                       int **firstneigh, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       const double cpu_time, bool &success);
double xndaf_gpu_bytes();                                               

/* ---------------------------------------------------------------------- */

PairXNDAFGPU::PairXNDAFGPU(LAMMPS *lmp) : PairXNDAFOMP(lmp), gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  reinitflag = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

PairXNDAFGPU::~PairXNDAFGPU()
{
  xndaf_gpu_clear();
}


void PairXNDAFGPU::compute(int eflag, int vflag)
{
  int i,j;
  ev_init(eflag, vflag);

  int nall = atom->nlocal + atom->nghost;
  int inum, host_start;

  bool success = true;
  int *ilist, *numneigh, **firstneigh;

  // calc sq and generate force table;
  if(ncall%update_interval==0){
    compute_sq();
    generateForceTable();
    #ifndef XNDAF_INSTANT_FORCE
    error->all(FLERR,"XNDAF_INSTANT_FORCE is not defined! Recompile!");
    #else
    xndaf_gpu_sendTable(frc);
    #endif
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


  if (gpu_mode != GPU_FORCE) {
    double sublo[3],subhi[3];
    if (domain->triclinic == 0) {
      sublo[0] = domain->sublo[0];
      sublo[1] = domain->sublo[1];
      sublo[2] = domain->sublo[2];
      subhi[0] = domain->subhi[0];
      subhi[1] = domain->subhi[1];
      subhi[2] = domain->subhi[2];
    } else {
      domain->bbox(domain->sublo_lamda,domain->subhi_lamda,sublo,subhi);
    }
    inum = atom->nlocal;
    firstneigh = xndaf_gpu_compute_n(neighbor->ago, inum, nall, atom->x,
                                     atom->type, sublo, subhi,
                                     atom->tag, atom->nspecial, atom->special,
                                     eflag, vflag, eflag_atom, vflag_atom,
                                     host_start, &ilist, &numneigh, cpu_time,
                                     success);
  } else {
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    xndaf_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
                      ilist, numneigh, firstneigh, eflag, vflag, eflag_atom,
                      vflag_atom, host_start, cpu_time, success);
  }
  if (!success)
    error->one(FLERR,"Insufficient memory on accelerator");

  if (host_start<inum) {
    cpu_time = platform::walltime();
    cpu_compute(host_start, inum, eflag, vflag, ilist, numneigh, firstneigh);
    cpu_time = platform::walltime() - cpu_time;
  }

  eng_vdwl=localerg;
}

void PairXNDAFGPU::init_style()
{
  ncall=0;
  // Repeat cutsq calculation because done after call to init_style
  if (!allocated) error->all(FLERR,"All pair coeffs are not set");
  double cell_size = r_max + neighbor->skin;

  int maxspecial=0;
  if (atom->molecular != Atom::ATOMIC)
    maxspecial=atom->maxspecial;
  int mnf = 5e-2 * neighbor->oneatom;
  int success = xndaf_gpu_init(atom->ntypes+1, nbin_r, r_max*r_max, ddr,
                             force->special_lj, atom->nlocal,
                             atom->nlocal+atom->nghost, mnf, maxspecial,
                             cell_size, gpu_mode, screen);
  GPU_EXTRA::check_flag(success,error,world);

  if (gpu_mode == GPU_FORCE) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
  }
  // else error->all(FLERR,"gpu_mode != GPU_FORCE is not supported yet!");
}

void PairXNDAFGPU::compute_sq()
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
    evalsq_gpu(ifrom, ito, thr, cnt_omp_private[tid]);

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
#if defined(_OPENMP)
#pragma omp parallel for private(src)
#endif
  for(int qk=0;qk<nbin_q;qk++){
    ssq[qk][1]=ssq[qk][2]=0;
    for (int gk=0;gk<npair;gk++){
      src=gk+3;
      ssq[qk][src]=0;
      for(int rk=0;rk<nbin_r;rk++){
        ssq[qk][src]+=(ggr[rk][gk]-1)*sinqr[qk][rk];
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

void PairXNDAFGPU::evalsq_gpu(int iifrom, int iito, ThrData * const thr, int **counter)
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
          counter[ibin][typ2pair[itype][jtype]]++;
        }
      }
    }
  }
}

void PairXNDAFGPU::cpu_compute(int start, int inum, int eflag, int /* vflag */,
                               int *ilist, int *numneigh, int **firstneigh) {
  int i,j,ii,jj,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,factor_lj;
  int *jlist,tbi;


  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  double *special_lj = force->special_lj;

  // loop over neighbors of my atoms
  const double cutsqall = cutsq[0][0];
  for (ii = start; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsqall) {
        rsq=sqrt(rsq);
        tbi=(int)(rsq/ddr);
        if(tbi>=nbin_r) continue;
        // fpair = frc[tbi][typ2pair[itype][jtype]]*factor_lj;
        fpair = getForce(tbi,typ2pair[itype][jtype])*factor_lj;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
      }
    }
  }
}