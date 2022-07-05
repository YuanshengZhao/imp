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

#include "pair_fmirl.h"

#include <cmath>

#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "math_const.h"
#include "domain.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairFMIRL::PairFMIRL(LAMMPS *lmp) : Pair(lmp),
nbin_r(0), npair(0), nfea(0), ncall(0), beta1t(1.0), beta2t(1.0), use_base(0)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute=1;
  // unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairFMIRL::~PairFMIRL()
{
  if (allocated) {
    memory->destroy(typ2pair);
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(u0);
    memory->destroy(f0_dr);
    memory->destroy(uf);
    memory->destroy(ff_dr);
    memory->destroy(ufi);
    memory->destroy(ffi_dr);
    memory->destroy(fea);
    memory->destroy(f_coef);
    memory->destroy(l2);
    memory->destroy(fea_true);
    memory->destroy(grad);
    memory->destroy(mon1);
    memory->destroy(mon2);
    memory->destroy(rr);
    memory->destroy(cnt);
    memory->destroy(cnt_all);
    if(use_base) memory->destroy(base);
    delete[] feout;
    delete[] grout;
  }
}

/* ---------------------------------------------------------------------- */

void PairFMIRL::compute(int eflag, int vflag)
{
  // if(comm->me==0) utils::logmesg(lmp,"call compute\n");
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz,fpair, evdwl;
  double rsq, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh, tbi,ptemp;

  ev_init(eflag, vflag);
  if(ncall%update_interval==0){
    compute_gr();
    generateForceTable();
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


  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  const double cutsqall = cutsq[0][0];

  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
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
      rsq = delx * delx + dely * dely + delz * delz;
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
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) {
          evdwl = u0[ptemp][tbi]*factor_lj+uf[ptemp][tbi];
        }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairFMIRL::settings(int narg, char **/*arg*/)
{
  // if(comm->me==0) utils::logmesg(lmp,"call settings\n");
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
// pair_coeff * * data_file init_file base learing_rate adam_epsilon update_interval output_interval fe_out gr_out
void PairFMIRL::coeff(int narg, char **arg)
{
  // if(comm->me==0) utils::logmesg(lmp,"call coeff\n");
  if (allocated) error->all(FLERR,"Cannot call a second pair_coeff");;

  ntypes = atom->ntypes;
  int dimension = domain->dimension;
  int triclinic = domain->triclinic;

  // Checking errors
  if (dimension == 2)
     error->all(FLERR,"FMIRL does not work with 2d structures");
  if (narg != 11)
     error->all(FLERR,"Illegal FMIRL Command");
  if (triclinic == 1)
     error->all(FLERR,"FMIRL does not work with triclinic structures");

  use_base=utils::inumeric(FLERR,arg[4],false,lmp);
  if(comm->me==0)
  {
    if(use_base) utils::logmesg(lmp,"use_base = T\n");
    else utils::logmesg(lmp,"use_base = F\n");
  }
  lr=utils::numeric(FLERR,arg[5],false,lmp);
  epsilon=utils::numeric(FLERR,arg[6],false,lmp);
  update_interval=utils::inumeric(FLERR,arg[7],false,lmp);
  output_interval=utils::inumeric(FLERR,arg[8],false,lmp);

  feout=new char [ 1+strlen(arg[9])];
  sprintf(feout,"%s",arg[9]);
  grout=new char [ 1+strlen(arg[10])];
  sprintf(grout,"%s",arg[10]);
  binout=new char [ 1+strlen(arg[3])];
  sprintf(binout,"%s",arg[3]);

  npair=ntypes*(ntypes+1)/2;
  memory->create(typ2pair,ntypes+1,ntypes+1,"rmdf:typ2pair");
  memory->create(setflag,ntypes+1,ntypes+1,"pair:setflag"); // must be set to avoid segmentation error
  memory->create(cutsq,ntypes+1,ntypes+1,"pair:cutsq");     // must be set to avoid segmentation error

  int iarg=0;
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      typ2pair[i][j]=typ2pair[j][i] = iarg++;
  read_file(arg[2],arg[3]);
  init_norm();
  allocated=1;
  for (int i = 0; i <= ntypes; i++) //cutsq[0][0] will be used. 
    for (int j = 0; j <= ntypes; j++) 
    {
      cutsq[i][j]=r_max*r_max;
      setflag[i][j] = 1;
    }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairFMIRL::init_style()
{
  // if(comm->me==0) utils::logmesg(lmp,"call init_style\n");
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style FMIRL requires atom IDs");
  ncall=0;
  int irequest = neighbor->request(this,instance_me);
  // neighbor->requests[irequest]->half = 1;
  // neighbor->requests[irequest]->full = 0;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairFMIRL::init_one(int i, int j)
{
  // if(comm->me==0) utils::logmesg(lmp,"call init_one\n");
  if (!allocated) error->all(FLERR,"All pair coeffs are not set");
  return r_max;
}

/* ---------------------------------------------------------------------- */
// binary file format : nfea, r_max, nbin_r, n_type, u0_0_0...u0_[npair-1]_[nbin_r-1], uf__...uf__, f0__...f0__, ff__...ff__,fea_true_...,l2__
void PairFMIRL::read_file(char *file, char *file_i)
{
  // if(comm->me==0) utils::logmesg(lmp,"call read_file\n");
  // open file on proc 0
  FILE *fp;
  if (comm->me == 0) {
    fp=fopen(file,"rb");
    if(!fp) {error->all(FLERR,"Data file not found"); return;}

    if(fread(&nfea,sizeof(int),1,fp)!=1){ error->all(FLERR,"Error reading nfea"); return; }
    else {utils::logmesg(lmp,"nfea = {}\n",nfea);}

    if(fread(&r_max,sizeof(double),1,fp)!=1){ error->all(FLERR,"Error reading r_max"); return; }
    else {utils::logmesg(lmp,"r_max = {}\n",r_max);}

    if(fread(&nbin_r,sizeof(int),1,fp)!=1){ error->all(FLERR,"Error reading nbin_r"); return; }
    else {utils::logmesg(lmp,"nbin_r = {}\n",nbin_r);}
    
    if(fread(&ntypes,sizeof(int),1,fp)!=1){ error->all(FLERR,"Error reading ntypes"); return; }
    else {utils::logmesg(lmp,"ntypes = {}\n",ntypes);}
    if(ntypes!=atom->ntypes) { error->all(FLERR,"Wrong ntypes"); return; }
  }

  MPI_Bcast(&r_max, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&nfea,  1, MPI_INT   , 0, world);
  MPI_Bcast(&nbin_r,1, MPI_INT   , 0, world);
  // ntypes is already retrived from atom->ntypes
  ddr=r_max/nbin_r;

  memory->create(u0,npair,nbin_r,"fmirl:u0");
  memory->create(f0_dr,npair,nbin_r,"fmirl:f0_dr");
  memory->create(uf,npair,nbin_r,"fmirl:uf");
  memory->create(ff_dr,npair,nbin_r,"fmirl:ff_dr");
  memory->create(ufi,nfea,npair,nbin_r,"fmirl:ufi");
  memory->create(ffi_dr,nfea,npair,nbin_r,"fmirl:ffi_dr");
  memory->create(fea,nfea,"fmirl:fea");
  memory->create(f_coef,nfea,"fmirl:f_coef");
  memory->create(l2,nfea,"fmirl:l2");
  memory->create(fea_true,nfea,"fmirl:fea_true");
  memory->create(grad,nfea,"fmirl:grad");
  memory->create(mon1,nfea,"fmirl:mon1");
  memory->create(mon2,nfea,"fmirl:mon2");
  memory->create(rr,nbin_r,"fmirl:rr");
  memory->create(cnt,npair,nbin_r,"fmirl:gnm");
  memory->create(cnt_all,npair,nbin_r,"fmirl:gnm");
  if(use_base) memory->create(base,nfea,"fmirl:base");


  for(int i=0;i<nbin_r;++i)
    rr[i]=r_max*(i+.5)/nbin_r;

  for(int i=0;i<nfea;++i)
    mon1[i]=mon2[i]=0;
  if (comm->me == 0) {
    for(int p=0;p<npair;++p)
      if(fread(u0[p],sizeof(double),nbin_r,fp)!=nbin_r){ error->all(FLERR,"Error reading u0"); return; }
    for(int f=0;f<nfea;++f)
      for(int p=0;p<npair;++p)
        if(fread(ufi[f][p],sizeof(double),nbin_r,fp)!=nbin_r){ error->all(FLERR,"Error reading ufi"); return; }

    for(int p=0;p<npair;++p)
      if(fread(f0_dr[p],sizeof(double),nbin_r,fp)!=nbin_r){ error->all(FLERR,"Error reading f0_dr"); return; }
    for(int f=0;f<nfea;++f)
      for(int p=0;p<npair;++p)
        if(fread(ffi_dr[f][p],sizeof(double),nbin_r,fp)!=nbin_r){ error->all(FLERR,"Error reading ffi_dr"); return; }

    if(fread(fea_true,sizeof(double),nfea,fp)!=nfea){ error->all(FLERR,"Error reading fea_true"); return; }
    if(fread(l2,sizeof(double),nfea,fp)!=nfea){ error->all(FLERR,"Error reading l2"); return; }

    fclose(fp);

    fp=fopen(file_i,"rb");
    if(!fp) {error->all(FLERR,"Init file not found"); return;}
    if(fread(f_coef,sizeof(double),nfea,fp)!=nfea){ error->all(FLERR,"Error reading fea_true"); return; }
    fclose(fp);
  }

  MPI_Bcast(u0[0],        npair*nbin_r,       MPI_DOUBLE, 0, world);
  MPI_Bcast(f0_dr[0],     npair*nbin_r,       MPI_DOUBLE, 0, world);
  MPI_Bcast(ufi[0][0],    nfea*npair*nbin_r,  MPI_DOUBLE, 0, world);
  MPI_Bcast(ffi_dr[0][0], nfea*npair*nbin_r,  MPI_DOUBLE, 0, world);
  MPI_Bcast(fea_true   ,  nfea,               MPI_DOUBLE, 0, world);
  MPI_Bcast(l2   ,        nfea,               MPI_DOUBLE, 0, world);
  MPI_Bcast(f_coef,       nfea,               MPI_DOUBLE, 0, world);

}

/* ---------------------------------------------------------------------- */

void PairFMIRL::init_norm()
{
  // if(comm->me==0) utils::logmesg(lmp,"call init_norm\n");
  // count atoms of each type that are also in group

  const int nlocal = atom->nlocal;
  // const int * const mask = atom->mask;
  const int * const type = atom->type;
  int *typecount = new int[ntypes+1];
  int *scratch = new int[ntypes+1];

  for (int ii = 1; ii <= ntypes; ii++) typecount[ii] = 0;
  for (int ii = 0; ii < nlocal; ii++)
    // if (mask[i] & groupbit) 
      ++typecount[type[ii]];

  natoms=0;
  MPI_Allreduce(typecount,scratch,ntypes+1,MPI_INT,MPI_SUM,world);
  for (int ii = 0; ii < ntypes; ii++) {
    typecount[ii] = scratch[ii+1];
    natoms+=typecount[ii];
    if(comm->me==0) utils::logmesg(lmp,"<FMIRL> N[{}] = {}\n",ii+1, typecount[ii]);
  }

  if(use_base){
    double *r2=new double[nbin_r];
    for(int ir=0;ir<nbin_r;++ir){
      r2[ir]=rr[ir]*rr[ir];
    }
    double *nor_c=new double[nfea];
    int ifea=0;
    for(int ii=0;ii<ntypes;++ii){
      for(int jj=ii;jj<ntypes;++jj){
        // ifea is used as ipair here!
        nor_c[ifea++]=typecount[ii]*(ii==jj? .5*(typecount[jj]-1):(double)(typecount[jj]))/natoms*ddr*MY_4PI;
      }
    }
    double accum;
    for(ifea=0;ifea<nfea;++ifea){
      base[ifea]=0;
      for(int ipa=0;ipa<npair;++ipa){
        accum=0;
        for(int ir=0;ir<nbin_r;++ir){
          accum+=ufi[ifea][ipa][ir]*r2[ir];
        }
        accum*=nor_c[ipa];
        base[ifea]+=accum;
      }
    }
    delete[] r2;
    delete[] nor_c;
  }

  delete[] typecount;
  delete[] scratch;
  if(comm->me==0) utils::logmesg(lmp,"norm generated\n");
}

void PairFMIRL::compute_gr()
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
          if (newton_pair || j < nlocal) cnt[typ2pair[itype][jtype]][ibin]+=2;
          else ++cnt[typ2pair[itype][jtype]][ibin];
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

void PairFMIRL::generateForceTable()
{  
  beta1t*=.9;
  beta2t*=.999;
  for(int fk=0;fk<nfea;++fk){
    grad[fk]=fea[fk]-fea_true[fk]-l2[fk]*f_coef[fk];
    mon1[fk]=mon1[fk]*.9+grad[fk]*.1;
    mon2[fk]=mon2[fk]*.999+grad[fk]*grad[fk]*.001;
    f_coef[fk]+=lr*mon1[fk]/(1-beta1t)/(epsilon+sqrt(mon2[fk]/(1-beta2t)));
  }
  for(int i=0;i<npair;++i){
    for(int rk=0;rk<nbin_r;++rk){
      ff_dr[i][rk]=uf[i][rk]=0;
    }
  }

  for(int fk=0;fk<nfea;++fk){
    for(int i=0;i<npair;++i){
      for(int rk=0;rk<nbin_r;rk++){
        uf[i][rk]+=f_coef[fk]*ufi[fk][i][rk];
        ff_dr[i][rk]+=f_coef[fk]*ffi_dr[fk][i][rk];
      }
    }
  }
}