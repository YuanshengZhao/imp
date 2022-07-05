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

#include "pair_xndaf.h"
#include "compute_xrd_consts.h"

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
#ifdef XNDAF_DEBUG
#include <chrono>
#include <iostream>
#endif

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairXNDAF::PairXNDAF(LAMMPS *lmp) : Pair(lmp),
nbin_r(0), nbin_q(0), npair(0)
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

PairXNDAF::~PairXNDAF()
{
  if (allocated) {
    memory->destroy(ssq);
    memory->destroy(iiq);
    memory->destroy(sinqr);
    memory->destroy(sff);
    memory->destroy(ggr);
    memory->destroy(cnt);
    memory->destroy(cnt_all);
    memory->destroy(typ2pair);
    memory->destroy(gnm);
    memory->destroy(frc);
    memory->destroy(frc_allocated);
    memory->destroy(neu_b);
    memory->destroy(sffn);
    memory->destroy(sqex);
    memory->destroy(wt);
    memory->destroy(kq);
    memory->destroy(mq);
    memory->destroy(wk);
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(sff_w);
    memory->destroy(sffn_w);
    memory->destroy(dsicqr_dr_div_r);
    memory->destroy(force_qspace);
  delete[] ztype;
  delete[] sqout;
  delete[] grout;
  }
}

/* ---------------------------------------------------------------------- */

void PairXNDAF::compute(int eflag, int vflag)
{
  // if(comm->me==0) utils::logmesg(lmp,"call compute\n");
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz,fpair;
  double rsq, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh, tbi;

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
  #ifdef XNDAF_DEBUG
  auto t1=std::chrono::high_resolution_clock::now();
  #endif
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    // utils::logmesg(lmp,"x{} {} {} {}\n",i,x[i][0],x[i][1],x[i][2]);
    // utils::logmesg(lmp,"f{} {} {} {}\n",i,f[i][0],f[i][1],f[i][2]);
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
        // fpair = frc[tbi][typ2pair[itype][jtype]]*factor_lj;
        fpair = getForce(tbi,typ2pair[itype][jtype])*factor_lj;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        // if (eflag) {
        //   evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
        //   evdwl *= factor_lj;
        // }

        // if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
  #ifdef XNDAF_DEBUG
  auto t2=std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "total compute loop " << time_span.count() << " s\n";
  #endif
  // if (vflag_fdotr) virial_fdotr_compute();
  eng_vdwl=localerg;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairXNDAF::settings(int narg, char **/*arg*/)
{
  // if(comm->me==0) utils::logmesg(lmp,"call settings\n");
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
// pair_coeff * * <sqex_file> <typs> <neutron_bs> nbin_r r_max nbin_q qmax factorx factorn update_interval output_interval sq_out gr_out
void PairXNDAF::coeff(int narg, char **arg)
{
  // if(comm->me==0) utils::logmesg(lmp,"call coeff\n");
  if (allocated) error->all(FLERR,"Cannot call a second pair_coeff");;

  ntypes = atom->ntypes;
  int dimension = domain->dimension;
  int triclinic = domain->triclinic;

  // Checking errors
  if (dimension == 2)
     error->all(FLERR,"Compute SQXF does not work with 2d structures");
  if (narg != 13+ntypes*2)
     error->all(FLERR,"Illegal Compute SQXF Command");
  if (triclinic == 1)
     error->all(FLERR,"Compute SQXF does not work with triclinic structures");

  // Define atom types for atomic scattering factor coefficients
  int iarg = 3;
  ztype = new int[ntypes];
  for (int i = 0; i < ntypes; i++) {
    ztype[i] = XRDmaxType + 1;
  }
  for (int i = 0; i < ntypes; i++) {
    for (int j = 0; j < XRDmaxType; j++) {
      if (utils::lowercase(arg[iarg]) == utils::lowercase(XRDtypeList[j])) {
        ztype[i] = j;
       }
     }
    if (ztype[i] == XRDmaxType + 1)
        error->all(FLERR,"Compute SQXF: Invalid ASF atom type");
    iarg++;
  }
  memory->create(neu_b,ntypes,"rdf:neu_b");
  for (int i = 0; i < ntypes; i++) {
    neu_b[i]=utils::numeric(FLERR,arg[iarg++],false,lmp);
    if(comm->me==0) utils::logmesg(lmp,"<SQXF> neutron_b[{}] = {}\n",i+1,neu_b[i]);
  }
  nbin_r = utils::inumeric(FLERR,arg[iarg],false,lmp);
  r_max = utils::numeric(FLERR,arg[iarg+1],false,lmp);
  nbin_q = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
  q_max = utils::numeric(FLERR,arg[iarg+3],false,lmp);
  ddr=r_max/nbin_r;
  factorx = utils::numeric(FLERR,arg[iarg+4],false,lmp);
  factorn = utils::numeric(FLERR,arg[iarg+5],false,lmp);
  update_interval=utils::inumeric(FLERR,arg[iarg+6],false,lmp);
  output_interval=utils::inumeric(FLERR,arg[iarg+7],false,lmp);

  sqout=new char [ 1+strlen(arg[iarg+8])];
  sprintf(sqout,"%s",arg[iarg+8]);
  grout=new char [ 1+strlen(arg[iarg+9])];
  sprintf(grout,"%s",arg[iarg+9]);

  npair=ntypes*(ntypes+1)/2;
  memory->create(ssq,nbin_q,3+npair,"rmdf:ssq");
  memory->create(iiq,nbin_q,2,"rmdf:iiq");
  memory->create(sinqr,nbin_q,nbin_r,"rmdf:sinqr");
  memory->create(dsicqr_dr_div_r,nbin_q,nbin_r,"rmdf:dsinqr_dr");
  memory->create(force_qspace,npair,nbin_q,"rmdf:force_qspace");
  memory->create(sff,nbin_q,npair,"rmdf:sff");
  memory->create(sff_w,nbin_q,npair,"rmdf:sff_w");
  memory->create(ggr,nbin_r,npair,"rmdf:ggr");
  memory->create(cnt,nbin_r,npair,"rmdf:cnt");
  memory->create(cnt_all,nbin_r,npair,"rmdf:cnt_all");
  memory->create(typ2pair,ntypes+1,ntypes+1,"rmdf:typ2pair");
  memory->create(gnm,nbin_r,npair,"rmdf:gnm");
  memory->create(frc,nbin_r,npair,"rmdf:frc");
  memory->create(frc_allocated,nbin_r,npair,"rmdf:frc_allocated");
  memory->create(sffn,npair,"rmdf:sffn");
  memory->create(sffn_w,npair,"rmdf:sffn_w");
  memory->create(wt,nbin_q,2,"rmdf:wt");
  memory->create(kq,nbin_q,2,"rmdf:kq");
  memory->create(mq,nbin_q,2,"rmdf:mq");
  memory->create(wk,nbin_q,2,"rmdf:wk");
  memory->create(sqex,nbin_q,2,"rmdf:sqex");
  memory->create(setflag,ntypes+1,ntypes+1,"pair:setflag"); // must be set to avoid segmentation error
  memory->create(cutsq,ntypes+1,ntypes+1,"pair:cutsq");     // must be set to avoid segmentation error

  iarg=0;
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      typ2pair[i][j]=typ2pair[j][i] = iarg++;
  read_file(arg[2]);
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

void PairXNDAF::init_style()
{
  // if(comm->me==0) utils::logmesg(lmp,"call init_style\n");
  // if (atom->tag_enable == 0)
    // error->all(FLERR,"Pair style XNDAF requires atom IDs");
  ncall=0;
  int irequest = neighbor->request(this,instance_me);
  // neighbor->requests[irequest]->half = 0;
  // neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairXNDAF::init_one(int i, int j)
{
  // if(comm->me==0) utils::logmesg(lmp,"call init_one\n");
  if (!allocated) error->all(FLERR,"All pair coeffs are not set");
  return r_max;
}

/* ---------------------------------------------------------------------- */
// file format : q sqx wtx kx mx sqn wtn kn mn
void PairXNDAF::read_file(char *file)
{
  // if(comm->me==0) utils::logmesg(lmp,"call read_file\n");
  // open file on proc 0
  if (comm->me == 0) {
    FILE *fp=fopen(file,"r");
    if(!fp){
      error->all(FLERR,"Exp S(Q) file not found");
      return;
    }
    utils::logmesg(lmp,"Reading {} rows\n",nbin_q);
    double sumx=0,sumn=0;
    for(int i=0;i<nbin_q;i++){
      if(fscanf(fp,"%*f %lf %lf %lf %lf %lf %lf %lf %lf\n",&sqex[i][0],&wt[i][0],&kq[i][0],&mq[i][0],
                                                           &sqex[i][1],&wt[i][1],&kq[i][1],&mq[i][1]) != 8){
        error->all(FLERR,"Error reading sqex");
        return;
      }
      sumx+=wt[i][0];
      sumn+=wt[i][1];
    }
    for(int i=0;i<nbin_q;i++){
      wt[i][0]/=sumx;
      wt[i][1]/=sumn;
      wk[i][0]=wt[i][0]*kq[i][0]*factorx*2;
      wk[i][1]=wt[i][1]*kq[i][1]*factorn*2;
    }
    fclose(fp);
    utils::logmesg(lmp,"Exp S(Q) final data {} {}\n",sqex[nbin_q-1][0],sqex[nbin_q-1][1]);
    // transfer exp sq to iq and normalize
    sumx=sumn=0;
    for(int i=0;i<nbin_q;i++){
      sumx+=(sqex[i][0]=sqex[i][0]*kq[i][0]+mq[i][0])*wt[i][0];
      sumn+=(sqex[i][1]=sqex[i][1]*kq[i][1]+mq[i][1])*wt[i][1];
    }
    for(int i=0;i<nbin_q;i++){
      sqex[i][0]-=sumx;
      sqex[i][1]-=sumn;
    }
    sumx=sumn=0;
    for(int i=0;i<nbin_q;i++){
      sumx+=sqex[i][0]*sqex[i][0]*wt[i][0];
      sumn+=sqex[i][1]*sqex[i][1]*wt[i][1];
    }
    sumx=sqrt(sumx);
    sumn=sqrt(sumn);
    for(int i=0;i<nbin_q;i++){
      sqex[i][0]/=sumx;
      sqex[i][1]/=sumn;
    }

  }

  MPI_Bcast(sqex[0], 2*nbin_q, MPI_DOUBLE, 0, world);
  MPI_Bcast(wt[0], 2*nbin_q, MPI_DOUBLE, 0, world);
  MPI_Bcast(kq[0], 2*nbin_q, MPI_DOUBLE, 0, world);
  MPI_Bcast(mq[0], 2*nbin_q, MPI_DOUBLE, 0, world);
  MPI_Bcast(wk[0], 2*nbin_q, MPI_DOUBLE, 0, world);
}

/* ---------------------------------------------------------------------- */

void PairXNDAF::init_norm()
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
      typecount[type[ii]]++;

  natoms=0;
  MPI_Allreduce(typecount,scratch,ntypes+1,MPI_INT,MPI_SUM,world);
  for (int ii = 0; ii < ntypes; ii++) {
    typecount[ii] = scratch[ii+1];
    if(neu_b[ii]!=0.0) natoms+=typecount[ii];
    if(comm->me==0) utils::logmesg(lmp,"<SQXF> N[{}] = {}\n",ii+1,neu_b[ii]!=0.0 ? typecount[ii] : -typecount[ii]);
  }

  double *f = new double[ntypes],qo4p,rj,dr=r_max/nbin_r,sffa;
  int pair_id;
  const double fourPiRho=MY_4PI;//*natoms/(domain->xprd)/(domain->yprd)/(domain->zprd);
  for (int qk=0;qk<nbin_q;qk++){
    // calc transform MX
    ssq[qk][0]=qo4p=q_max*(qk+.5)/nbin_q;
    for(int rk=0;rk<nbin_r;rk++){
      rj=r_max*(rk+.5)/nbin_r;
      sffa=qo4p*rj;
      sinqr[qk][rk]=sin(sffa)*fourPiRho*rj/qo4p*dr;
      dsicqr_dr_div_r[qk][rk]=(cos(sffa)*sffa-sin(sffa))/(sffa*rj*rj);
      // sinqr[qk][rk]*=sin(MY_2PI*(rk+.5)/nbin_r)/(MY_2PI*(rk+.5)/nbin_r); //Lanczos window
    }
    // if(comm->me==0) utils::logmesg(lmp,"sinqr finished\n");
    //calc X weight
    qo4p/=MY_4PI;
    for (int ii = 0; ii < ntypes; ii++) {
      f[ii] = ASFXRD[ztype[ii]][8];
      for (int C = 0; C < 8 ; C+=2) {
        f[ii] += ASFXRD[ztype[ii]][C] * exp(-1 * ASFXRD[ztype[ii]][C+1] * qo4p * qo4p );
      }
    }
    pair_id=0;
    sffa=0;
    for (int ii = 0; ii < ntypes; ii++) {
      for (int jj = ii; jj < ntypes; jj++) {
        sff[qk][pair_id]=(ii==jj? 1 : 2)*f[ii]*f[jj]*typecount[ii]*typecount[jj]/natoms/natoms;
        sffa+=sff[qk][pair_id++];
      }
    }
    for (int ii = 0; ii < npair; ii++) {
        sff[qk][ii]/=sffa;
    }    
    pair_id=0;
    for (int ii = 0; ii < ntypes; ii++) {
      for (int jj = ii; jj < ntypes; jj++) {
        sff_w[qk][pair_id++]=f[ii]*f[jj]/sffa;
      }
    }
    // if(comm->me==0) utils::logmesg(lmp,"sff finished\n");
  }
  //calc N weight
  pair_id=0;
  sffa=0;
  for (int ii = 0; ii < ntypes; ii++) {
    for (int jj = ii; jj < ntypes; jj++) {
      sffn[pair_id]=(ii==jj? 1 : 2)*neu_b[ii]*neu_b[jj]*typecount[ii]*typecount[jj]/natoms/natoms;
      sffa+=sffn[pair_id++];
    }
  }
  for (int ii = 0; ii < npair; ii++) {
      sffn[ii]/=sffa;
  }    
  pair_id=0;
  for (int ii = 0; ii < ntypes; ii++) {
    for (int jj = ii; jj < ntypes; jj++) {
      sffn_w[pair_id++]=neu_b[ii]*neu_b[jj]/sffa;
    }
  }
  // if(comm->me==0) utils::logmesg(lmp,"sffn finished\n");
  //norm form gr
  // qo4p=(domain->xprd)*(domain->yprd)*(domain->zprd);
  for(int rk=0;rk<nbin_r;rk++){
    rj=(pow(r_max*(rk+1)/nbin_r,3)-pow(r_max*(rk)/nbin_r,3))/3;
    pair_id=0;
    for (int ii = 0; ii < ntypes; ii++) {
      for (int jj = ii; jj < ntypes; jj++) {
        gnm[rk][pair_id++]=.5/(rj)*2/(MY_4PI)/(typecount[ii]*(ii==jj? (typecount[jj]-1) : 2*typecount[jj]))*natoms;//*qo4p;
      }
    }
  }
  // if(comm->me==0) utils::logmesg(lmp,"gnm finished\n");

  delete[] f;
  delete[] typecount;
  delete[] scratch;
  if(comm->me==0) utils::logmesg(lmp,"norm generated\n");
}

void PairXNDAF::compute_sq()
{
  #ifdef XNDAF_DEBUG
  auto t1=std::chrono::high_resolution_clock::now();
  #endif
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
      cnt[j][i] = 0;

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
          if (newton_pair || j < nlocal) cnt[ibin][typ2pair[itype][jtype]]+=2;
          else ++cnt[ibin][typ2pair[itype][jtype]];
        }
      }

    }
  }

  // sum histograms across procs

  MPI_Allreduce(cnt[0],cnt_all[0],npair*nbin_r,MPI_INT,MPI_SUM,world);

  // convert counts to g(r) and coord(r) and copy into output array
  // vfrac = fraction of volume in shell m
  // npairs = number of pairs, corrected for duplicates
  // duplicates = pairs in which both atoms are the same

  for (int gk=0;gk<npair;gk++){
    for(int rk=0;rk<nbin_r;rk++){
      ggr[rk][gk]=cnt_all[rk][gk]*gnm[rk][gk];
      // array[rk][1+gk]=ggr[rk][gk];
    }
  }

  // gr -> sq
  const double vinv=natoms/((domain->xprd)*(domain->yprd)*(domain->zprd));
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

// lazy evaluation of force table.
double PairXNDAF::getForce(int idx_r, int idx_pair)
{
  #ifdef XNDAF_INSTANT_FORCE
  return frc[idx_r][idx_pair];
  #else
  // if(std::isinf(frc[idx_r][idx_pair]))
  if(frc_allocated[idx_r][idx_pair]) return frc[idx_r][idx_pair];

  double ffc=0;
  for(int qk=0;qk<nbin_q;qk++){
    ffc+=force_qspace[idx_pair][qk]*dsicqr_dr_div_r[qk][idx_r];
  }
  frc[idx_r][idx_pair]=ffc;
  frc_allocated[idx_r][idx_pair]=1;
  return ffc;
  // return ffc;
  #endif
}

void PairXNDAF::generateForceTable()
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
  for(int qk=0;qk<nbin_q;qk++){
    diffx=(sqex[qk][0]-iiq[qk][0]*crsx)*wk[qk][0]/normx;
    diffn=(sqex[qk][1]-iiq[qk][1]*crsn)*wk[qk][1]/normn;
    for(int i=0;i<npair;i++){
      force_qspace[i][qk]=diffx*sff_w[qk][i]+diffn*sffn_w[i];
      #ifdef XNDAF_INSTANT_FORCE
      for(int rk=0;rk<nbin_r;rk++){
        frc[rk][i]+=force_qspace[i][qk]*dsicqr_dr_div_r[qk][rk];
        // frc_allocated[rk][i]=1;
      }
      #endif
    }
  }
  localerg=((1-crsx*normx)*factorx+(1-crsn*normn)*factorn)*atom->nlocal;

  #ifdef XNDAF_DEBUG
  auto t2=std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "force table " << time_span.count() << " s\n";
  #endif
}