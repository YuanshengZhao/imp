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
PairStyle(table/compute,PairTBCOMP);
// clang-format on
#else

#ifndef LMP_PAIR_TABLE_COMPUTE_H
#define LMP_PAIR_TABLE_COMPUTE_H

// #define XNDAF_DEBUG
#define XNDAF_INSTANT_FORCE

#include "pair.h"

namespace LAMMPS_NS {

class PairTBCOMP : public Pair {
 public:
  PairTBCOMP(class LAMMPS *);
  virtual ~PairTBCOMP();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  protected:

  Compute *force_compute;
  double force_cutoff,force_cutoff_sq;
  int update_interval,output_interval; // interval for updating sq and output
  bigint ncall;
  int **typ2pair;       //typ to pair
  double **frc;                   // force divded by r
  int nbin_r;                // number of bins
  double ddr,r_max;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
