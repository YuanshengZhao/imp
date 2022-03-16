/***************************************************************************
                                    xndaf.h
                             -------------------
                            Trung Dac Nguyen (ORNL)

  Class for acceleration of the xndaf/cut pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : nguyentd@ornl.gov
 ***************************************************************************/

#ifndef LAL_XNDAF_H
#define LAL_XNDAF_H

#include "lal_base_atomic.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class XNDAF : public BaseAtomic<numtyp, acctyp> {
 public:
  XNDAF();
  ~XNDAF();

  /// Clear any previous data and set up for a new LAMMPS run
  /** \param max_nbors initial number of rows in the neighbor matrix
    * \param cell_size cutoff + skin
    * \param gpu_split fraction of particles handled by device
    *
    * Returns:
    * -  0 if successful
    * - -1 if fix gpu not found
    * - -3 if there is an out of memory error
    * - -4 if the GPU library was not compiled for GPU
    * - -5 Double precision is not supported on card **/
  int init(const int ntypes, const int ntable, double host_cutsq,
           double host_dr, double *host_special_lj,
           const int nlocal, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           const double gpu_split, FILE *screen);
  void sendTable(double **tb);

  /// Clear all host and device data
  /** \note This is called at the beginning of the init() routine **/
  void clear();

  /// Returns memory usage on device per atom
  int bytes_per_atom(const int max_nbors) const;

  /// Total host memory used by library for pair style
  double host_memory_usage() const;

  // --------------------------- TYPE DATA --------------------------

  /// cutsq and dr
  numtyp cutsq;
  numtyp drinv;

  UCL_D_Vec<int> tableindex;
  /// force table
  UCL_D_Vec<numtyp> frc_tb;
  /// Special lj values
  UCL_D_Vec<numtyp> sp_lj;

  int n_table,n_pair;

  /// If atom type constants fit in shared memory, use fast kernels
  bool shared_types;

  /// Number of atom types
  int _lj_types;

 private:
  bool _allocated;
  int loop(const int eflag, const int vflag);
};

}

#endif
