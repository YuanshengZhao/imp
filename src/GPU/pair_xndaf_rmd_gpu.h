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
PairStyle(xndaf/rmd/gpu,PairXNDAFRMDGPU);
// clang-format on
#else

#ifndef LMP_PAIR_XNDAFRMDGPU_H
#define LMP_PAIR_XNDAFRMDGPU_H

#include "pair_xndaf_gpu.h"

namespace LAMMPS_NS {

class PairXNDAFRMDGPU : public PairXNDAFGPU { // use omp version of compute sq to accelerate.
 public:
  PairXNDAFRMDGPU(class LAMMPS *);
 protected:
  virtual void read_file(char *) override;
  virtual void generateForceTable() override;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
