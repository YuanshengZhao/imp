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
   Contributing authors: Shawn Coleman & Douglas Spearot (Arkansas)
   Updated: 06/17/2015-2
------------------------------------------------------------------------- */

#include "compute_xndaf_rmd_force.h"
#include "compute_xrd_consts.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "math_const.h"
#include "memory.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "force.h"
#include "pair.h"
#include <iostream>

#include <cmath>
#include <cstring>

#include "omp_compat.h"
using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */
// usage: compute <id> all sqxf <sqex_file> <typs> <neutron_bs> nbin_r r_max nbin_q qmax factorx factorn output_interval sq_out gr_out[group is ignored and set to all
ComputeXNDAFRMDFORCE::ComputeXNDAFRMDFORCE(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), ztype(nullptr), sinqr(nullptr), sff(nullptr),
   ggr(nullptr), cnt(nullptr), cnt_all(nullptr), typ2pair(nullptr), gnm(nullptr), neu_b(nullptr), sffn(nullptr), ncall(0)
{

  int ntypes = atom->ntypes;
  int dimension = domain->dimension;
  int triclinic = domain->triclinic;

  // Checking errors
  if (dimension == 2)
     error->all(FLERR,"Compute SQXF does not work with 2d structures");
  if (narg != 13+ntypes*2)
     error->all(FLERR,"Illegal Compute SQXF Command");
  if (triclinic == 1)
     error->all(FLERR,"Compute SQXF does not work with triclinic structures");

  array_flag = 1;
  extarray = 0;

  // Define atom types for atomic scattering factor coefficients
  int iarg = 4;
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
  r_max_sq = r_max*r_max;
  nbin_q = utils::inumeric(FLERR,arg[iarg+2],false,lmp);
  q_max = utils::numeric(FLERR,arg[iarg+3],false,lmp);
  ddr = r_max/nbin_r;
  factorx = utils::numeric(FLERR,arg[iarg+4],false,lmp);
  factorn = utils::numeric(FLERR,arg[iarg+5],false,lmp);
  output_interval = utils::inumeric(FLERR,arg[iarg+6],false,lmp);

  sqout=new char [ 1+strlen(arg[iarg+7])];
  sprintf(sqout,"%s",arg[iarg+7]);
  grout=new char [ 1+strlen(arg[iarg+8])];
  sprintf(grout,"%s",arg[iarg+8]);

  size_array_cols = npair = ntypes*(ntypes+1)/2;
  size_array_rows = nbin_r;

  memory->create(ssq,nbin_q,3+npair,"rmdf:ssq");
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
  memory->create(array,nbin_r,npair,"rmdf:array");
  memory->create(frc_allocated,nbin_r,npair,"rmdf:frc_allocated");
  memory->create(sffn,npair,"rmdf:sffn");
  memory->create(sffn_w,npair,"rmdf:sffn_w");
  memory->create(wt,nbin_q,2,"rmdf:wt");
  memory->create(kq,nbin_q,2,"rmdf:kq");
  memory->create(mq,nbin_q,2,"rmdf:mq");
  memory->create(wk,nbin_q,2,"rmdf:wk");
  memory->create(sqex,nbin_q,2,"rmdf:sqex");

  array[1][0]=r_max;
  iarg=0;
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      typ2pair[i][j]=typ2pair[j][i] = iarg++;
  read_file(arg[3]);
  init_norm();
}

/* ---------------------------------------------------------------------- */

ComputeXNDAFRMDFORCE::~ComputeXNDAFRMDFORCE()
{
  memory->destroy(array);
  memory->destroy(sinqr);
  memory->destroy(sff);
  memory->destroy(ggr);
  memory->destroy(cnt);
  memory->destroy(cnt_all);
  memory->destroy(typ2pair);
  memory->destroy(gnm);
  memory->destroy(neu_b);
  memory->destroy(sffn);
  delete[] ztype;
}

void ComputeXNDAFRMDFORCE::init()
{
  double skin = neighbor->skin,mycutneigh;
  mycutneigh = r_max + skin;
  double cutghost;            // as computed by Neighbor and Comm
  if (force->pair)
    cutghost = MAX(force->pair->cutforce+skin,comm->cutghostuser);
  else
    cutghost = comm->cutghostuser;
  if (mycutneigh > cutghost)
    error->all(FLERR,"Compute rdf cutoff exceeds ghost atom range - "
               "use comm_modify cutoff command");
  if (force->pair && mycutneigh < force->pair->cutforce + skin)
    if (comm->me == 0)
      error->warning(FLERR,"Compute rdf cutoff less than neighbor cutoff - "
                     "forcing a needless neighbor list build");

  init_norm();

  // need an occasional half neighbor list
  // if user specified, request a cutoff = cutoff_user + skin
  // skin is included b/c Neighbor uses this value similar
  //   to its cutneighmax = force cutoff + skin
  // also, this NeighList may be used by this compute for multiple steps
  //   (until next reneighbor), so it needs to contain atoms further
  //   than cutoff_user apart, just like a normal neighbor list does

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->occasional = 1;
  neighbor->requests[irequest]->cut = 1;
  neighbor->requests[irequest]->cutoff = mycutneigh;
  neighbor->requests[irequest]->half = 1;
  neighbor->requests[irequest]->full = 0;
}

void ComputeXNDAFRMDFORCE::init_list(int /*id*/, NeighList *ptr)
{
  // std::cout<<"compute list is"<<ptr<<std::endl;
  // std::cout<<"modify is"<<modify<<std::endl;
  list = ptr;
}

/* ---------------------------------------------------------------------- */
// file format : q sqx wtx kx mx sqn wtn kn mn
void ComputeXNDAFRMDFORCE::read_file(char *file)
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
    fclose(fp);
    for(int i=0;i<nbin_q;i++){
      wt[i][0]=wt[i][0]/sumx*factorx*2;
      wt[i][1]=wt[i][1]/sumn*factorn*2;
      // iiq[i][0]=iiq[i][1]=0;
    }
    utils::logmesg(lmp,"Exp S(Q) final data {} {}\n",sqex[nbin_q-1][0],sqex[nbin_q-1][1]);

  }

  MPI_Bcast(sqex[0], 2*nbin_q, MPI_DOUBLE, 0, world);
  MPI_Bcast(wt[0], 2*nbin_q, MPI_DOUBLE, 0, world);
}


/* ---------------------------------------------------------------------- */
void ComputeXNDAFRMDFORCE::init_norm()
{
  // if(comm->me==0) utils::logmesg(lmp,"call init_norm\n");
  // count atoms of each type that are also in group

  const int nlocal = atom->nlocal;
  const int ntypes = atom->ntypes;
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
        f[ii] += ASFXRD[ztype[ii]][C] * exp(-ASFXRD[ztype[ii]][C+1] * qo4p * qo4p );
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
    rj=(1 + 3*rk * (1 + rk))/3.0 * std::pow(r_max/nbin_r,3);  //(pow(r_max*(rk+1)/nbin_r,3)-pow(r_max*(rk)/nbin_r,3))/3;
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


void ComputeXNDAFRMDFORCE::compute_sq()
{
  #ifdef XNDAF_DEBUG
  auto t1=std::chrono::high_resolution_clock::now();
  #endif
  // if(comm->me==0) utils::logmesg(lmp,"call compute_sq\n");
  int src,inum,jnum,i,j,ii,jj,itype,jtype,ibin;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double xtmp,ytmp,ztmp,delx,dely,delz,r;
  
  //calc gr
  neighbor->build_one(list);

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
      if (r < r_max_sq)
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

void ComputeXNDAFRMDFORCE::generateForceTable()
{  
  #ifdef XNDAF_DEBUG
  auto t1=std::chrono::high_resolution_clock::now();
  #endif

  for(int i=0;i<npair;i++){
    for(int rk=2;rk<nbin_r;rk++){
      array[rk][i]=0;
    }
  }

  double localerg=0;
  double diffx,diffn;
  for(int qk=0;qk<nbin_q;qk++){
    // note: ssq[][0] is Q
    diffx=(sqex[qk][0]-ssq[qk][1]);
    diffn=(sqex[qk][1]-ssq[qk][2]);
    localerg+=(diffx*diffx)*wt[qk][0]+(diffn*diffn)*wt[qk][1];
    diffx*=wt[qk][0];
    diffn*=wt[qk][1];
    for(int i=0;i<npair;i++){
      force_qspace[i][qk]=diffx*sff_w[qk][i]+diffn*sffn_w[i];
      for(int rk=2;rk<nbin_r;rk++){
        array[rk][i]+=force_qspace[i][qk]*dsicqr_dr_div_r[qk][rk];
      }
    }
  }
  **array=localerg*(.5*atom->nlocal);
  
  #ifdef XNDAF_DEBUG
  auto t2=std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "force table " << time_span.count() << " s\n";
  #endif
}

void ComputeXNDAFRMDFORCE::compute_array()
{
  compute_sq();
  generateForceTable();

    if(comm->me==0 && (ncall++)%output_interval==0){
    // utils::logmesg(lmp,"output sq and gr\n");
    FILE *fp=fopen(sqout,"w");
    for(int i=0;i<nbin_q;i++)
    {
      fprintf(fp,"%lf %lf %lf",ssq[i][0],ssq[i][1],ssq[i][2]);
      for(int j=0;j<npair;j++){
        fprintf(fp," %lf",ssq[i][3+j]);
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
    fp=fopen(grout,"w");
    for(int i=0;i<nbin_r;i++)
    {
      fprintf(fp,"%lf",(i+.5)*ddr);
      for(int j=0;j<npair;j++){
        fprintf(fp," %lf",ggr[i][j]);
      }
      fprintf(fp,"\n");
    }
    fclose(fp);
  }
}