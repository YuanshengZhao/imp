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

#include "pair_table_compute.h"
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
#include "modify.h"
#include "compute.h"
#include <iostream>
#ifdef XNDAF_DEBUG
#include <chrono>
#endif

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairTBCOMP::PairTBCOMP(LAMMPS *lmp) : Pair(lmp),
nbin_r(0)
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

PairTBCOMP::~PairTBCOMP()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    modify->delete_compute(force_compute -> id);
  }
}

/* ---------------------------------------------------------------------- */

void PairTBCOMP::compute(int eflag, int vflag)
{
  // if(comm->me==0) utils::logmesg(lmp,"call compute\n");
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;
  double rsq, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh, tbi;

  ev_init(eflag, vflag);
  // calc sq and generate force table;
  if(ncall%update_interval==0){
    force_compute -> compute_array();
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
  // const double cutsqall = force_cutoff_sq;// cutsq[0][0];

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

      if (rsq < force_cutoff_sq) {
        rsq=sqrt(rsq);
        tbi=(int)(rsq/ddr);
        if(tbi>=nbin_r) continue;
        fpair = frc[tbi][typ2pair[itype][jtype]]*factor_lj;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

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
  if (eflag) {
    eng_vdwl=**frc;
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTBCOMP::settings(int narg, char **/*arg*/)
{
  // if(comm->me==0) utils::logmesg(lmp,"call settings\n");
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
// pair_coeff * * force_cutoff update_interval force_compute_args
// the compute must compute array with structure
/* it is assumed that the first 2 rows will not be used (too small r)
erg   -  -  ...
r_max -  -  ...
f1    f2 f3 ...
.
.
.
*/
void PairTBCOMP::coeff(int narg, char **arg)
{
  // if(comm->me==0) utils::logmesg(lmp,"call coeff\n");
  if (allocated) error->all(FLERR,"Cannot call a second pair_coeff");

  int ntypes = atom->ntypes;

  if (narg < 7)
     error->all(FLERR,"Illegal Pair table/compute Command");

  force_compute = modify -> add_compute(narg-4,arg+4);
  
  force_cutoff = utils::numeric(FLERR,arg[2],false,lmp);
  force_cutoff_sq = force_cutoff*force_cutoff;
  update_interval=utils::inumeric(FLERR,arg[3],false,lmp);

  if (force_compute->size_array_cols != ntypes*(ntypes+1)/2) error->all(FLERR,"Compute SQXF: wrong size_array_cols");
  if (force_compute->size_array_rows < 3) error->all(FLERR,"Compute SQXF: size_array_rows too small");

  nbin_r = force_compute -> size_array_rows;
  frc = force_compute->array;
  r_max = frc[1][0];
  if (r_max < force_cutoff) error->all(FLERR,"Compute SQXF: table_r_max < force_cutoff");
  ddr = r_max/nbin_r;
  if(comm->me==0) utils::logmesg(lmp,"nbin_r = {}, r_max = {}, force_cutoff = {}\n", nbin_r, r_max, force_cutoff);

  memory->create(typ2pair,ntypes+1,ntypes+1,"rmdf:typ2pair");
  memory->create(setflag, ntypes+1,ntypes+1,"pair:setflag");   // must be set to avoid segmentation error
  memory->create(cutsq,   ntypes+1,ntypes+1,"pair:cutsq");     // must be set to avoid segmentation error

  int iarg=0;
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      typ2pair[i][j]=typ2pair[j][i] = iarg++;
  allocated=1;
  for (int i = 0; i <= ntypes; i++) //cutsq[0][0] will be used. 
    for (int j = 0; j <= ntypes; j++) 
    {
      cutsq[i][j] = force_cutoff_sq;
      setflag[i][j] = 1;
    }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTBCOMP::init_style()
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

double PairTBCOMP::init_one(int i, int j)
{
  // if(comm->me==0) utils::logmesg(lmp,"call init_one\n");
  if (!allocated) error->all(FLERR,"All pair coeffs are not set");
  return force_cutoff;
}
