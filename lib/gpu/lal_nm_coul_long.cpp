/***************************************************************************
                               nm_coul_long.cpp
                             -------------------
                            W. Michael Brown (ORNL)

  Class for acceleration of the nm/cut/coul/long pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : brownw@ornl.gov
 ***************************************************************************/

#if defined(USE_OPENCL)
#include "nm_coul_long_cl.h"
#elif defined(USE_CUDART)
const char *nm_coul_long=0;
#else
#include "nm_coul_long_cubin.h"
#endif

#include "lal_nm_coul_long.h"
#include <cassert>
namespace LAMMPS_AL {
#define NMCoulLongT NMCoulLong<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
NMCoulLongT::NMCoulLong() : BaseCharge<numtyp,acctyp>(),
                                    _allocated(false) {
}

template <class numtyp, class acctyp>
NMCoulLongT::~NMCoulLong() {
  clear();
}

template <class numtyp, class acctyp>
int NMCoulLongT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int NMCoulLongT::init(const int ntypes,
                           double **host_cutsq, double **host_gma,
                           double **host_fnm, double **host_nn, double **host_mm,
                           double **host_r02, double **host_offset,
                           double *host_special_lj, const int nlocal,
                           const int nall, const int max_nbors,
                           const int maxspecial, const double cell_size,
                           const double gpu_split, FILE *_screen,
                           double **host_cut_ljsq, const double host_cut_coulsq,
                           double *host_special_coul, const double qqrd2e,
                           const double g_ewald) {
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                            _screen,nm_coul_long,"k_nm_coul_long");
  if (success!=0)
    return success;

  // If atom type constants fit in shared memory use fast kernel
  int lj_types=ntypes;
  shared_types=false;
  int max_shared_types=this->device->max_shared_types();
  if (lj_types<=max_shared_types && this->_block_size>=max_shared_types) {
    lj_types=max_shared_types;
    shared_types=true;
  }
  _lj_types=lj_types;

  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(lj_types*lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<lj_types*lj_types; i++)
    host_write[i]=0.0;

  nm1.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,nm1,host_write,host_gma,host_fnm,
           host_cutsq, host_cut_ljsq);

  nm3.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  this->atom->type_pack4(ntypes,lj_types,nm3,host_write,host_nn,host_mm,
                         host_offset, host_r02);

  sp_lj.alloc(8,*(this->ucl_device),UCL_READ_ONLY);
  for (int i=0; i<4; i++) {
    host_write[i]=host_special_lj[i];
    host_write[i+4]=host_special_coul[i];
  }
  ucl_copy(sp_lj,host_write,8,false);

  _cut_coulsq=host_cut_coulsq;
  _qqrd2e=qqrd2e;
  _g_ewald=g_ewald;

  _allocated=true;
  this->_max_bytes=nm1.row_bytes()+nm3.row_bytes()+sp_lj.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void NMCoulLongT::reinit(const int ntypes, double **host_cutsq, double **host_gma,
                           double **host_fnm, double **host_nn, double **host_mm,
                           double **host_r02,
                         double **host_offset, double **host_cut_ljsq) {
  // Allocate a host write buffer for data initialization
  UCL_H_Vec<numtyp> host_write(_lj_types*_lj_types*32,*(this->ucl_device),
                               UCL_WRITE_ONLY);

  for (int i=0; i<_lj_types*_lj_types; i++)
    host_write[i]=0.0;

  this->atom->type_pack4(ntypes,_lj_types,nm1,host_write,host_gma,host_fnm,
           host_cutsq, host_cut_ljsq);
  this->atom->type_pack4(ntypes,_lj_types,nm3,host_write,host_nn,host_mm,
                         host_offset, host_r02);
}

template <class numtyp, class acctyp>
void NMCoulLongT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  nm1.clear();
  nm3.clear();
  sp_lj.clear();
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double NMCoulLongT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(NMCoulLong<numtyp,acctyp>);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int NMCoulLongT::loop(const int eflag, const int vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (shared_types) {
    this->k_pair_sel->set_size(GX,BX);
    this->k_pair_sel->run(&this->atom->x, &nm1, &nm3, &sp_lj,
                          &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &eflag,
                          &vflag, &ainum, &nbor_pitch, &this->atom->q,
                          &_cut_coulsq, &_qqrd2e, &_g_ewald,
                          &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x, &nm1, &nm3,
                     &_lj_types, &sp_lj, &this->nbor->dev_nbor,
                     &this->_nbor_data->begin(), &this->ans->force,
                     &this->ans->engv, &eflag, &vflag, &ainum,
                     &nbor_pitch, &this->atom->q, &_cut_coulsq,
                     &_qqrd2e, &_g_ewald, &this->_threads_per_atom);
  }
  this->time_pair.stop();
  return GX;
}

template class NMCoulLong<PRECISION,ACC_PRECISION>;
}
