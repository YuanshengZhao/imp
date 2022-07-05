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

#include "pair_fmirl_gpu.h"

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

using namespace LAMMPS_NS;

int fmirl_gpu_init(const int ntypes, const int ntable, double host_cutsq,
           double host_dr, double *host_special_lj,
           const int inum, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           int &gpu_mode, FILE *screen);
void fmirl_gpu_clear();
void fmirl_gpu_sendTable(double **host_u0,double **host_f0,double **host_uf,double **host_ff);
int ** fmirl_gpu_compute_n(const int ago, const int inum_full,
                           const int nall, double **host_x, int *host_type,
                           double *sublo, double *subhi, tagint *tag, int **nspecial,
                           tagint **special, const bool eflag, const bool vflag,
                           const bool eatom, const bool vatom, int &host_start,
                           int **ilist, int **jnum, const double cpu_time,
                           bool &success);
void fmirl_gpu_compute(const int ago, const int inum_full, const int nall,
                       double **host_x, int *host_type, int *ilist, int *numj,
                       int **firstneigh, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       const double cpu_time, bool &success);
double fmirl_gpu_bytes();                                               

/* ---------------------------------------------------------------------- */

PairFMIRLGPU::PairFMIRLGPU(LAMMPS *lmp) : PairFMIRL(lmp), gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  reinitflag = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

PairFMIRLGPU::~PairFMIRLGPU()
{
  fmirl_gpu_clear();
}


void PairFMIRLGPU::compute(int eflag, int vflag)
{
  int i,j;
  ev_init(eflag, vflag);

  int nall = atom->nlocal + atom->nghost;
  int inum, host_start;

  bool success = true;
  int *ilist, *numneigh, **firstneigh;

  // calc sq and generate force table;
  if(ncall%update_interval==0){
    compute_gr();
    generateForceTable();
    fmirl_gpu_sendTable(u0,f0_dr,uf,ff_dr);
  }
  if(comm->me==0 && ncall%output_interval==0 && ncall>=output_interval){
    // utils::logmesg(lmp,"output sq and gr\n");
    FILE *fp=fopen(feout,"a+");
    for(i=0;i<nfea;i++)
      fprintf(fp,"%lf %lf %lf\n",fea[i],f_coef[i],grad[i]);
    fclose(fp);
    fp=fopen(binout,"wb");
    fwrite(f_coef,sizeof(double),nfea,fp);
    fclose(fp);
    fp=fopen(grout,"w");
    for(i=0;i<nbin_r;i++)
    {
      fprintf(fp,"%lf",rr[i]);
      for(j=0;j<npair;j++){
        fprintf(fp," %d %e %e %e %e",cnt_all[j][i],u0[j][i],f0_dr[j][i],uf[j][i],ff_dr[j][i]);
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
  }

  ++ncall;


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
    firstneigh = fmirl_gpu_compute_n(neighbor->ago, inum, nall, atom->x,
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
    fmirl_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
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

}

void PairFMIRLGPU::init_style()
{
  ncall=0;
  // Repeat cutsq calculation because done after call to init_style
  if (!allocated) error->all(FLERR,"All pair coeffs are not set");
  double cell_size = r_max + neighbor->skin;

  int maxspecial=0;
  if (atom->molecular != Atom::ATOMIC)
    maxspecial=atom->maxspecial;
  int mnf = 5e-2 * neighbor->oneatom;
  int success = fmirl_gpu_init(atom->ntypes+1, nbin_r, r_max*r_max, ddr,
                             force->special_lj, atom->nlocal,
                             atom->nlocal+atom->nghost, mnf, maxspecial,
                             cell_size, gpu_mode, screen);
  GPU_EXTRA::check_flag(success,error,world);

  if (gpu_mode == GPU_FORCE) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
  }
  else error->all(FLERR,"gpu_mode != GPU_FORCE is not supported yet!");
}

void PairFMIRLGPU::compute_gr()
{
  // if(comm->me==0) utils::logmesg(lmp,"call compute_sq\n");
  int src,inum,jnum,i,j,ii,jj,itype,jtype,ibin;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double xtmp,ytmp,ztmp,delx,dely,delz,r;
  
  //calc gr
  // neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  int newton_pair = force->newton_pair;
  int nlocal = atom->nlocal;

  // zero the histogram counts

  for (i = 0; i < npair; i++)
    for (j = 0; j < nbin_r; j++)
      cnt[i][j] = 0;

  // tally the RDF
  // both atom i and j must be in fix group
  // itype,jtype must have been specified by user
  // consider I,J as one interaction even if neighbor pair is stored on 2 procs
  // tally I,J pair each time I is central atom, and each time J is central

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  const double cutsqall = cutsq[0][0];

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    // if (!(mask[i] & groupbit)) continue;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      // if (!(mask[j] & groupbit)) continue;
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      // if(jtype<itype) continue;


      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      r = delx*delx + dely*dely + delz*delz;
      if (r < cutsqall)
      {
        r=sqrt(r);
        ibin = static_cast<int> (r/ddr);
        if (ibin < nbin_r) {
          ++cnt[typ2pair[itype][jtype]][ibin];
        }
      }

    }
  }

  // sum histograms across procs

  MPI_Allreduce(cnt[0],cnt_all[0],npair*nbin_r,MPI_INT,MPI_SUM,world);

  for (int fk=0;fk<nfea;++fk){
    fea[fk]=0;
    for (int gk=0;gk<npair;++gk){
      for(int rk=0;rk<nbin_r;++rk){
        fea[fk]+=cnt_all[gk][rk]*ufi[fk][gk][rk];
      }
    }
    fea[fk]/=(2*natoms);
  }

  if(use_base)
  {
    double vinv=1.0/((domain->xprd)*(domain->yprd)*(domain->zprd));
    for (int fk=0;fk<nfea;++fk) fea[fk]-=base[fk]*vinv;
  }

}

void PairFMIRLGPU::cpu_compute(int start, int inum, int eflag, int /* vflag */,
                               int *ilist, int *numneigh, int **firstneigh) {
  int i,j,ii,jj,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair,evdwl;
  double rsq,factor_lj;
  int *jlist,tbi,ptemp;


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
        ptemp=typ2pair[itype][jtype];
        fpair = f0_dr[ptemp][tbi]*factor_lj+ff_dr[ptemp][tbi];

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;

        if (eflag) {
          evdwl = u0[ptemp][tbi]*factor_lj+uf[ptemp][tbi];
        }

        if (evflag) ev_tally_full(i, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
}