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
ComputeStyle(sqxd,ComputeSQXD);
// clang-format on
#else

#ifndef LMP_COMPUTE_SQXD_H
#define LMP_COMPUTE_SQXD_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSQXD : public Compute {
 public:
  ComputeSQXD(class LAMMPS *, int, char **);
  ~ComputeSQXD();
  void init();
  void compute_array();

 private:
  int npair;
  int *ztype;           // Atomic number of the different atom types
  bigint natoms;
  double q_max;
  int *nq_bin,*q_bin,n_bin;             // Num qs in bin, q to bin, n_bin
  double **qs,**cosqr,**sinqr,**cosqr_all,**sinqr_all;
  int nq;                // number of qs
  double *neu_b,*sffn,limn;         // neutron scattering length
  double **sff,*limx;            // weight of each partial sq, high q limit
};

}    // namespace LAMMPS_NS

#endif
#endif
