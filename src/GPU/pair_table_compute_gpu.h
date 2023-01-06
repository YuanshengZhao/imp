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
PairStyle(table/compute/gpu,PairTBCOMPGPU);
// clang-format on
#else

#ifndef LMP_PAIR_TABLE_COMPUTE_GPU_H
#define LMP_PAIR_TABLE_COMPUTE_GPU_H

#include "pair_table_compute.h"

namespace LAMMPS_NS {

class PairTBCOMPGPU : public PairTBCOMP {
 public:
  PairTBCOMPGPU(class LAMMPS *);
  ~PairTBCOMPGPU();
  void cpu_compute(int, int, int, int, int *, int *, int **);
  virtual void compute(int, int) override;
  void init_style() override;
  enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
