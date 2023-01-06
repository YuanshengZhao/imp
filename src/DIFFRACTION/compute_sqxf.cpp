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

#include "compute_sqxf.h"
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
// usage: compute <id> all sqxf <typs> <neutron_bs> nbin_r r_max nbin_q qmax [group is ignored and set to all
ComputeSQXF::ComputeSQXF(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), ztype(nullptr), sinqr(nullptr), sff(nullptr),
   ggr(nullptr), cnt(nullptr), cnt_all(nullptr), typ2pair(nullptr), gnm(nullptr),neu_b(nullptr),sffn(nullptr)
{

  int ntypes = atom->ntypes;
  int dimension = domain->dimension;
  int triclinic = domain->triclinic;

  // Checking errors
  if (dimension == 2)
     error->all(FLERR,"Compute SQXF does not work with 2d structures");
  if (narg != 7+ntypes*2)
     error->all(FLERR,"Illegal Compute SQXF Command");
  if (triclinic == 1)
     error->all(FLERR,"Compute SQXF does not work with triclinic structures");

  array_flag = 1;
  extarray = 0;

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
    if(comm->me==0) utils::logmesg(lmp,"<SQXF id:{}> neutron_b[{}] = {}\n", id,i+1,neu_b[i]);
  }
  nbin_r = utils::numeric(FLERR,arg[iarg],false,lmp);
  r_max = utils::numeric(FLERR,arg[iarg+1],false,lmp);
  nbin_q = utils::numeric(FLERR,arg[iarg+2],false,lmp);
  q_max = utils::numeric(FLERR,arg[iarg+3],false,lmp);
  ddr=r_max/nbin_r;

  size_array_rows = nbin_q;
  npair=ntypes*(ntypes+1)/2;
  size_array_cols = 3+npair;
  memory->create(array,size_array_rows,size_array_cols,"sqxf:array");
  memory->create(sinqr,nbin_q,nbin_r,"sqxf:sinqr");
  memory->create(sff,nbin_q,npair,"sqxf:sff");
  memory->create(ggr,nbin_r,npair,"sqxf:ggr");
  memory->create(cnt,nbin_r,npair,"sqxf:cnt");
  memory->create(cnt_all,nbin_r,npair,"sqxf:cnt_all");
  memory->create(typ2pair,ntypes+1,ntypes+1,"sqxf:typ2pair");
  memory->create(gnm,nbin_r,npair,"sqxf:gnm");
  memory->create(sffn,npair,"sqxf:sffn");

  iarg=0;
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      typ2pair[i][j]=typ2pair[j][i] = iarg++;
}

/* ---------------------------------------------------------------------- */

ComputeSQXF::~ComputeSQXF()
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

void ComputeSQXF::init()
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

void ComputeSQXF::init_list(int /*id*/, NeighList *ptr)
{
  // std::cout<<"compute list is"<<ptr<<std::endl;
  // std::cout<<"modify is"<<modify<<std::endl;
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void ComputeSQXF::init_norm()
{
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
      typecount[type[ii]]++;
      
  natoms=0;
  MPI_Allreduce(typecount,scratch,ntypes+1,MPI_INT,MPI_SUM,world);
  for (int ii = 0; ii < ntypes; ii++) {
    typecount[ii] = scratch[ii+1];
    if(neu_b[ii]!=0.0) natoms+=typecount[ii];
    if(comm->me==0) utils::logmesg(lmp,"<SQXF id:{}> N[{}] = {}\n", id,ii+1,neu_b[ii]!=0.0 ? typecount[ii] : -typecount[ii]);
  }

  double *f = new double[ntypes],qo4p,rj,dr=r_max/nbin_r,sffa;
  int pair_id;
  const double fourPiRho=MY_4PI;//*natoms/(domain->xprd)/(domain->yprd)/(domain->zprd);
  for (int qk=0;qk<nbin_q;qk++){
    // calc transform MX
    array[qk][0]=qo4p=q_max*(qk+.5)/nbin_q;
    for(int rk=0;rk<nbin_r;rk++){
      rj=r_max*(rk+.5)/nbin_r;
      sinqr[qk][rk]=sin(qo4p*rj)*fourPiRho*rj/qo4p*dr;
      // sinqr[qk][rk]*=sin(MY_2PI*(rk+.5)/nbin_r)/(MY_2PI*(rk+.5)/nbin_r); //Lanczos window
    }

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

  delete[] f;
  delete[] typecount;
  delete[] scratch;
}
/* ---------------------------------------------------------------------- */
// output array: q sqX sqN partial_sqs
void ComputeSQXF::compute_array()
{
  // printf("compute_array");
  invoked_array = update->ntimestep;
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
      r = sqrt(delx*delx + dely*dely + delz*delz);
      ibin = static_cast<int> (r/ddr);
      if (ibin < nbin_r) {
        if (newton_pair || j < nlocal) cnt[ibin][typ2pair[itype][jtype]]+=2;
        else cnt[ibin][typ2pair[itype][jtype]]++;
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
    array[qk][1]=array[qk][2]=0;
    for (int gk=0;gk<npair;gk++){
      src=gk+3;
      array[qk][src]=0;
      for(int rk=0;rk<nbin_r;rk++){
        array[qk][src]+=(ggr[rk][gk]-vinv)*sinqr[qk][rk];
      }
      array[qk][1]+=array[qk][src]*sff[qk][gk];
      array[qk][2]+=array[qk][src]*sffn[gk];
    }
  }
}
