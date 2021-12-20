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

#include "compute_sqxd.h"
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

#include <cmath>
#include <cstring>

#include "omp_compat.h"
using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */
// usage: compute <id> all sqxd <typs> <neutron_bs> qmax lim degen_threash [group is ignored and set to all
ComputeSQXD::ComputeSQXD(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), ztype(nullptr), sff(nullptr),
  qs(nullptr),cosqr(nullptr),sinqr(nullptr),cosqr_all(nullptr),sinqr_all(nullptr),
  neu_b(nullptr),sffn(nullptr),nq_bin(nullptr),q_bin(nullptr)
{

  int ntypes = atom->ntypes;
  int natoms = group->count(igroup);
  int dimension = domain->dimension;
  int triclinic = domain->triclinic;

  // Checking errors
  if (dimension == 2)
     error->all(FLERR,"Compute SQXD does not work with 2d structures");
  if (narg != 6+ntypes*2)
     error->all(FLERR,"Illegal Compute SQXD Command");
  if (triclinic == 1)
     error->all(FLERR,"Compute SQXD does not work with triclinic structures");

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
        error->all(FLERR,"Compute SQXD: Invalid ASF atom type");
    iarg++;
  }
  memory->create(neu_b,ntypes,"rdf:neu_b");
  for (int i = 0; i < ntypes; i++) {
    neu_b[i]=utils::numeric(FLERR,arg[iarg++],false,lmp);
    if(comm->me==0) utils::logmesg(lmp,"<SQXD id:{}> neutron_b[{}] = {}\n", id,i+1,neu_b[i]);
  }
  q_max = utils::numeric(FLERR,arg[iarg],false,lmp);

  //count qs
  nq=0;
  int lim=utils::numeric(FLERR,arg[iarg+1],false,lmp);
  double bx=MY_2PI/(domain->xprd),by=MY_2PI/(domain->yprd),bz=MY_2PI/(domain->zprd),qx,qy,qz;
  int g_mx=(int)(q_max/bx),g_my=(int)(q_max/by),g_mz=(int)(q_max/bz),h_my,h_mz;
  for(int ix=g_mx;ix>=0;ix--){
    qx=ix*bx;
    h_my=ix>lim? lim : g_my;
    for(int iy=h_my;iy>=0;iy--){
      qy=iy*by;
      h_mz=(ix>lim || iy>lim)? lim : g_mz;
      for(int iz=h_mz;iz>=0;iz--){
        if(ix==0 && iy==0 && iz==0) continue;
        qz=iz*bz;
        if(qx*qx+qy*qy+qz*qz<=q_max*q_max) nq++;
      }
    }
  }
  if (nq==0) error->all(FLERR,"Compute SQXD: No reciprocal vectors found");
  if(comm->me==0) utils::logmesg(lmp,"<SQXD id:{}> N_vectors = {}\n", id,nq);
  memory->create(qs,nq,3,"sqxd:qs");
  memory->create(q_bin,nq,"sqxd:q_bin");
  //generate qqs
  double qqs,*qq=new double[nq];
  int nq_back=nq;
  nq=0;
  for(int ix=g_mx;ix>=0;ix--){
    qx=ix*bx;
    h_my=ix>lim? lim : g_my;
    for(int iy=h_my;iy>=0;iy--){
      qy=iy*by;
      h_mz=(ix>lim || iy>lim)? lim : g_mz;
      for(int iz=h_mz;iz>=0;iz--){
        if(ix==0 && iy==0 && iz==0) continue;
        qz=iz*bz;
        qqs=qx*qx+qy*qy+qz*qz;
        if(qqs<=q_max*q_max) 
        {
          if(nq==nq_back){
            error->all(FLERR,"Compute SQXD: Reciprocal vectors mismatch");
            return;
          }
          qq[nq]=sqrt(qqs);
          qs[nq][0]=qx;
          qs[nq][1]=qy;
          qs[nq][2]=qz;
          nq++;
        }
      }
    }
  }
  if(nq!=nq_back){
    error->all(FLERR,"Compute SQXD: Reciprocal vectors mismatch");
    return;
  }
  //sort qqs
  //nq_back used as min_id
  for(int i=0;i<nq;i++){
    nq_back=i;
    qqs=qq[nq_back];
    for(int j=i+1;j<nq;j++){
      if(qq[j]<qqs){
        nq_back=j;
        qqs=qq[nq_back];
      }
    }
    if(nq_back!=i)
    {
      qqs=qq[nq_back];qq[nq_back]=qq[i];qq[i]=qqs;
      qqs=qs[nq_back][0];qs[nq_back][0]=qs[i][0];qs[i][0]=qqs;
      qqs=qs[nq_back][1];qs[nq_back][1]=qs[i][1];qs[i][1]=qqs;
      qqs=qs[nq_back][2];qs[nq_back][2]=qs[i][2];qs[i][2]=qqs;
    }
  }

  //bining qs
  double dq_thr=utils::numeric(FLERR,arg[iarg+2],false,lmp);
  int *nq_bin_temp=new int[nq];
  n_bin=0;
  int ibins=0,ibine;
  while(ibins<nq){
    qq[n_bin]=qqs=qq[ibins];
    nq_bin_temp[n_bin]=1;
    q_bin[ibins]=n_bin;
    ibine=ibins+1;
    while(ibine<nq && qq[ibine]-qqs<dq_thr){
      qq[n_bin]+=qq[ibine];
      nq_bin_temp[n_bin]++;
      q_bin[ibine]=n_bin;
      ibine++;
    }
    if(comm->me==0) utils::logmesg(lmp,"<SQXD id:{}> bin[{}] min {} max {} n {}\n", id,n_bin,qqs,qq[ibine-1],nq_bin_temp[n_bin]);
    n_bin++;
    ibins=ibine;
  }

  size_array_rows = n_bin;
  npair=ntypes*(ntypes+1)/2;
  size_array_cols = 3+npair;
  memory->create(array,size_array_rows,size_array_cols,"sqxd:array");
  memory->create(nq_bin,n_bin,"sqxd:nq_bin");
  for(ibins=0;ibins<n_bin;ibins++)
  {
    nq_bin[ibins]=nq_bin_temp[ibins];
    array[ibins][0]=qq[ibins]/nq_bin[ibins];
  }


  memory->create(sinqr,ntypes+1,nq,"sqxd:sinqr");
  memory->create(cosqr,ntypes+1,nq,"sqxd:cosqr");
  memory->create(sinqr_all,ntypes+1,nq,"sqxd:sinqr_all");
  memory->create(cosqr_all,ntypes+1,nq,"sqxd:cosqr_all");
  memory->create(sff,n_bin,npair,"sqxd:sff");
  memory->create(sffn,npair,"sqxd:sffn");

  delete[] qq;
  delete[] nq_bin_temp;
}

/* ---------------------------------------------------------------------- */

ComputeSQXD::~ComputeSQXD()
{
  memory->destroy(array);
  memory->destroy(sinqr);
  memory->destroy(cosqr);
  memory->destroy(sinqr_all);
  memory->destroy(cosqr_all);
  memory->destroy(qs);
  memory->destroy(sff);
  memory->destroy(sffn);
  memory->destroy(neu_b);
  memory->destroy(q_bin);
  memory->destroy(nq_bin);
  delete[] ztype;
}

/* ---------------------------------------------------------------------- */
void ComputeSQXD::init()
{
  // count atoms of each type that are also in group

  const int nlocal = atom->nlocal;
  const int ntypes = atom->ntypes;
  const int * const mask = atom->mask;
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
    natoms+=typecount[ii];
    if(comm->me==0) utils::logmesg(lmp,"<SQXD id:{}> N[{}] = {}\n", id,ii+1,typecount[ii]);
  }

  double *f = new double[ntypes],qo4p,sffa;
  int pair_id;
  for (int qk=0;qk<n_bin;qk++){
    //calc X weight
    qo4p=array[qk][0]/MY_4PI;
    for (int ii = 0; ii < ntypes; ii++) {
      f[ii] = ASFXRD[ztype[ii]][8];
      for (int C = 0; C < 8 ; C+=2) {
        f[ii] += ASFXRD[ztype[ii]][C] * exp(-1 * ASFXRD[ztype[ii]][C+1] * qo4p * qo4p );
      }
    }
    pair_id=0;
    sffa=0;
    for (int ii = 0; ii < ntypes; ii++) {
        sffa+=f[ii]*f[ii]*typecount[ii]/natoms;
    }   
    for (int ii = 0; ii < ntypes; ii++) {
      for (int jj = ii; jj < ntypes; jj++) {
        sff[qk][pair_id++]=(ii==jj? 1 : 2)*f[ii]*f[jj]/sffa;
      }
    } 
  }
  //calc N weight
  pair_id=0;
  sffa=0;
  for (int ii = 0; ii < ntypes; ii++) {
      sffa+=neu_b[ii]*neu_b[ii]*typecount[ii]/natoms;
  }   
  for (int ii = 0; ii < ntypes; ii++) {
    for (int jj = ii; jj < ntypes; jj++) {
      sffn[pair_id++]=(ii==jj? 1 : 2)*neu_b[ii]*neu_b[jj]/sffa;
    }
  }  

  delete[] f;
  delete[] typecount;
  delete[] scratch;
}
/* ---------------------------------------------------------------------- */
// output array: q sqX sqN partial_sqs
void ComputeSQXD::compute_array()
{
  invoked_array = update->ntimestep;

  const int nlocal = atom->nlocal;
  int ntypes = atom->ntypes;
  double **x = atom->x;
  int *type = atom->type;
  double qr;
  int typ;

  for(int gk=1;gk<=ntypes;gk++){
    for(int qk=0;qk<nq;qk++)
    sinqr[gk][qk]=cosqr[gk][qk]=0;
  }

  for(int i=0;i<nlocal;i++){
    typ=type[i];
    for(int qk=0;qk<nq;qk++){
      qr=qs[qk][0]*x[i][0]+qs[qk][1]*x[i][1]+qs[qk][2]*x[i][2];
      sinqr[typ][qk]+=sin(qr);
      cosqr[typ][qk]+=cos(qr);
    }
  }

  // sum across procs

  MPI_Allreduce(sinqr[1],sinqr_all[1],(ntypes)*nq,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(cosqr[1],cosqr_all[1],(ntypes)*nq,MPI_DOUBLE,MPI_SUM,world);

  // calc sq
  int pair_id;
  for(int qk=0;qk<n_bin;qk++){
    for(pair_id=1;pair_id<size_array_cols;pair_id++) array[qk][pair_id]=0;
  }
  for(int qk=0;qk<nq;qk++){
    pair_id=3;
    typ=q_bin[qk];
    for (int ii = 1; ii <= ntypes; ii++) {
      for (int jj = ii; jj <= ntypes; jj++) {
        array[typ][pair_id++]+=(cosqr_all[ii][qk]*cosqr_all[jj][qk]+sinqr_all[ii][qk]*sinqr_all[jj][qk]);
      }
    }
  }
  for(int qk=0;qk<n_bin;qk++){
    for(pair_id=3;pair_id<size_array_cols;pair_id++){ 
      array[qk][pair_id]/=(natoms*nq_bin[qk]);
      array[qk][1]+=array[qk][pair_id]*sff[qk][pair_id-3];
      array[qk][2]+=array[qk][pair_id]*sffn[pair_id-3];
    }
  }

}
