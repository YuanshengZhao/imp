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
PairStyle(xndaf/rmd/omp,PairXNDAFRMDOMP);
// clang-format on
#else

#ifndef LMP_PAIR_XNDAFRMD_OMP_H
#define LMP_PAIR_XNDAFRMD_OMP_H

#include "pair_xndaf_omp.h"
#include "thr_omp.h"

// #define XNDAF_OMP_DEBUG
// #define XNDAF_INSTANT_FORCE

namespace LAMMPS_NS {

class PairXNDAFRMDOMP : public PairXNDAFOMP {
 public:
  PairXNDAFRMDOMP(class LAMMPS *);

 protected:
  virtual void read_file(char *) override;
  virtual void generateForceTable() override;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
