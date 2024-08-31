/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(sqxf/rmd/force,ComputeXNDAFRMDFORCE);
// clang-format on
#else

#ifndef LMP_COMPUTE_XNDAF_RMD_FORCE_H
#define LMP_COMPUTE_XNDAF_RMD_FORCE_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeXNDAFRMDFORCE : public Compute {
 public:
  ComputeXNDAFRMDFORCE(class LAMMPS *, int, char **);
  ~ComputeXNDAFRMDFORCE();
  void init();
  void init_list(int, class NeighList *);
  void compute_array();

 private:
  int npair;
  int *ztype;           // Atomic number of the different atom types
  int **typ2pair;       //typ to pair
  double **ggr,**ssq,**sqex;        // gr
  int **cnt,**cnt_all;        // count sum over procs
  int nbin_r,nbin_q;                // number of bins
  double r_max,q_max,ddr,r_max_sq;
  double **sinqr,**dsicqr_dr_div_r;  // r,q,fourierMX
  double **force_qspace;
  double *neu_b,*sffn,*sffn_w;         // neutron scattering length
  double **sff,**gnm,**sff_w;            // weight of each partial sq, normalization of gr
  double **wt,**kq,**mq,**wk;                   // weight at each point, I(Q)=kq(Q) S(Q) + mq(Q), w*k*factor
//   double **frc;                   // force divded by r // stored in array
  int **frc_allocated;                   // force divded by r
  double factorx,factorn;               // over factor
  bigint natoms,ncall;
  char *sqout,*grout; // output file
  int update_interval,output_interval; // interval for updating sq and output

  class NeighList *list;    // half neighbor list
  void init_norm();
  virtual void read_file(char *);
  virtual void compute_sq();
  virtual void generateForceTable();
};

}    // namespace LAMMPS_NS

#endif
#endif
