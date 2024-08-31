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
ComputeStyle(sqxf,ComputeSQXF);
// clang-format on
#else

#ifndef LMP_COMPUTE_SQXF_H
#define LMP_COMPUTE_SQXF_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSQXF : public Compute {
 public:
  ComputeSQXF(class LAMMPS *, int, char **);
  ~ComputeSQXF();
  void init();
  void init_list(int, class NeighList *);
  void compute_array();

 private:
  int npair;
  int *ztype;           // Atomic number of the different atom types
  int **typ2pair;       //typ to pair
  double **ggr;        // gr
  int **cnt,**cnt_all;        // count sum over procs
  int nbin_r,nbin_q;                // number of bins
  double r_max,q_max,ddr;
  double **sinqr;  // r,q,fourierMX
  double *neu_b,*sffn;         // neutron scattering length
  double **sff,**gnm;            // weight of each partial sq, normalization of gr
  bigint natoms;

  class NeighList *list;    // half neighbor list
  void init_norm();
};

}    // namespace LAMMPS_NS

#endif
#endif
