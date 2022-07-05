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
PairStyle(xndaf,PairXNDAF);
// clang-format on
#else

#ifndef LMP_PAIR_XNDAF_H
#define LMP_PAIR_XNDAF_H

// #define XNDAF_DEBUG
#define XNDAF_INSTANT_FORCE

#include "pair.h"

namespace LAMMPS_NS {

class PairXNDAF : public Pair {
 public:
  PairXNDAF(class LAMMPS *);
  virtual ~PairXNDAF();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  double init_one(int, int);
  void init_style();

  protected:

  virtual void read_file(char *);
  int npair,ntypes,ncall;
  int *ztype;           // Atomic number of the different atom types
  int **typ2pair;       //typ to pair
  double **ggr,**ssq,**iiq,**sqex;        // gr
  int **cnt,**cnt_all;        // count sum over procs
  int nbin_r,nbin_q;                // number of bins
  double r_max,q_max,ddr;
  double **sinqr,**dsicqr_dr_div_r;  // r,q,fourierMX
  double **force_qspace;
  double *neu_b,*sffn,*sffn_w;         // neutron scattering length
  double **sff,**gnm,**sff_w;            // weight of each partial sq, normalization of gr
  double **wt,**kq,**mq,**wk;                   // weight at each point, I(Q)=kq(Q) S(Q) + mq(Q), w*k*factor
  double **frc;                   // force divded by r
  int **frc_allocated;                   // force divded by r
  double factorx,factorn;               // over factor
  bigint natoms;
  char *sqout,*grout; // output file
  int update_interval,output_interval; // interval for updating sq and output
  double localerg;
  void init_norm();
  virtual void compute_sq();
  virtual void generateForceTable();
  double getForce(int,int);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
