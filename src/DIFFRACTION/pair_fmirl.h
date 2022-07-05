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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(fmirl,PairFMIRL);
// clang-format on
#else

#ifndef LMP_PAIR_FMIRL_H
#define LMP_PAIR_FMIRL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairFMIRL : public Pair {
 public:
  PairFMIRL(class LAMMPS *);
  virtual ~PairFMIRL();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  protected:

  void read_file(char *,char*);
  int npair,ntypes;
  int nfea;
  double r_max,ddr,*rr;
  int nbin_r;
  double **u0,**f0_dr,**uf,**ff_dr;
  double ***ufi,***ffi_dr;
  double *fea,*f_coef,*l2,*fea_true;
  double *grad,*mon1,*mon2; 
  double lr;
  double beta1t,beta2t;
  int update_interval,output_interval;
  bigint ncall;
  int **typ2pair;
  double epsilon;
  int **cnt,**cnt_all;
  int use_base;
  double *base;

  char *grout,*feout,*binout;
  bigint natoms;
  void init_norm();
  virtual void compute_gr();
  virtual void generateForceTable();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
