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
PairStyle(xndaf/rmd,PairXNDAFRMD);
// clang-format on
#else

#ifndef LMP_PAIR_XNDAF_RMD_H
#define LMP_PAIR_XNDAF_RMD_H

#include "pair.h"
#include "pair_xndaf.h"

namespace LAMMPS_NS {

class PairXNDAFRMD : public PairXNDAF {
 public:
  PairXNDAFRMD(class LAMMPS *);
 protected:

  virtual void read_file(char *) override;
  virtual void generateForceTable() override;
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

*/
