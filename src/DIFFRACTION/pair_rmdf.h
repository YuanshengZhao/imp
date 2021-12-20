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
PairStyle(rmdf,PairRMDF);
// clang-format on
#else

#ifndef LMP_PAIR_RMDF_H
#define LMP_PAIR_RMDF_H

#include "pair.h"

namespace LAMMPS_NS {

class PairRMDF : public Pair {
 public:
  PairRMDF(class LAMMPS *);
  virtual ~PairRMDF();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  private:

  void read_file(char *);
  int npair,ntypes,ncall;
  int *ztype;           // Atomic number of the different atom types
  int **typ2pair;       //typ to pair
  double **ggr,**ssq,**sqex;        // gr
  int **cnt,**cnt_all;        // count sum over procs
  int nbin_r,nbin_q;                // number of bins
  double r_max,q_max,ddr;
  double **sinqr,**dsicqr_dr;  // r,q,fourierMX
  double *neu_b,*sffn,*sffn_w;         // neutron scattering length
  double **sff,**gnm,**sff_w;            // weight of each partial sq, normalization of gr
  double **wt;                   // weight at each point
  double **frc;                   // force
  double factor;               // over factor
  bigint natoms;
  char *sqout,*grout; // output file
  int update_interval,output_interval; // interval for updating sq and output
  void init_norm();
  void compute_sq();
  void generateForceTable();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
