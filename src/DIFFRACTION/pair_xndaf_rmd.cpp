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

#include "pair_xndaf_rmd.h"
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

PairXNDAFRMD::PairXNDAFRMD(LAMMPS *lmp) : PairXNDAF(lmp) {}

/* ---------------------------------------------------------------------- */
// file format : q sqx wtx kx mx sqn wtn kn mn
void PairXNDAFRMD::read_file(char *file)
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
      iiq[i][0]=iiq[i][1]=0;
    }
    utils::logmesg(lmp,"Exp S(Q) final data {} {}\n",sqex[nbin_q-1][0],sqex[nbin_q-1][1]);

  }

  MPI_Bcast(sqex[0], 2*nbin_q, MPI_DOUBLE, 0, world);
  MPI_Bcast(wt[0], 2*nbin_q, MPI_DOUBLE, 0, world);
}

void PairXNDAFRMD::generateForceTable()
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

  localerg=0;
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
      #ifdef XNDAF_INSTANT_FORCE
      for(int rk=0;rk<nbin_r;rk++){
        frc[rk][i]+=force_qspace[i][qk]*dsicqr_dr_div_r[qk][rk];
        // frc_allocated[rk][i]=1;
      }
      #endif
    }
  }
  localerg*=(.5*atom->nlocal);
  
  #ifdef XNDAF_DEBUG
  auto t2=std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << "force table " << time_span.count() << " s\n";
  #endif
}