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
PairStyle(xndaf/omp,PairXNDAFOMP);
// clang-format on
#else

#ifndef LMP_PAIR_XNDAFOMP_H
#define LMP_PAIR_XNDAFOMP_H

#include "pair_xndaf.h"
#include "thr_omp.h"

// #define XNDAF_OMP_DEBUG
// #define XNDAF_INSTANT_FORCE

namespace LAMMPS_NS {

class PairXNDAFOMP : public PairXNDAF, public ThrOMP {
 public:
  PairXNDAFOMP(class LAMMPS *);
  virtual void compute(int, int) override;
  virtual void coeff(int, char **) override;

 protected:
  int *** cnt_omp_private;
  virtual void compute_sq() override;
  virtual void generateForceTable() override;

  template <int NEWTON_PAIR>
  void eval(int ifrom, int ito, ThrData *const thr);
  template <int NEWTON_PAIR>
  void evalsq(int ifrom, int ito, ThrData *const thr, int **counter);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
