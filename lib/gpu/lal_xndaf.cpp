/***************************************************************************
                                   xndaf.cpp
                             -------------------
                            Trung Dac Nguyen (ORNL)

  Class for acceleration of the xndaf pair style.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : nguyentd@ornl.gov
 ***************************************************************************/

#ifdef USE_OPENCL
#include "xndaf_cl.h"
#elif defined(USE_CUDART)
const char *xndaf=0;
#else
#include "xndaf_cubin.h"
#endif

#include "lal_xndaf.h"
#include <cassert>
namespace LAMMPS_AL {
#define XNDAFT XNDAF<numtyp, acctyp>

extern Device<PRECISION,ACC_PRECISION> device;

template <class numtyp, class acctyp>
XNDAFT::XNDAF() : BaseAtomic<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
XNDAFT::~XNDAF() {
  clear();
}

template <class numtyp, class acctyp>
int XNDAFT::bytes_per_atom(const int max_nbors) const {
  return this->bytes_per_atom_atomic(max_nbors);
}

template <class numtyp, class acctyp>
int XNDAFT:: init(const int ntypes, const int ntable, double host_cutsq,
           double host_dr, double *host_special_lj,
           const int nlocal, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           const double gpu_split, FILE *screen) {
  int success;
  success=this->init_atomic(nlocal,nall,max_nbors,maxspecial,cell_size,gpu_split,
                            screen,xndaf,"k_xndaf");
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

  cutsq=(numtyp)host_cutsq;
  drinv=(numtyp)(1.0/host_dr);
  // Allocate a host write buffer for data initialization
  n_table=ntable;
  UCL_H_Vec<int> host_write_int(lj_types*lj_types,*(this->ucl_device),
                               UCL_WRITE_ONLY);
  tableindex.alloc(lj_types*lj_types,*(this->ucl_device),UCL_READ_ONLY);
  int tbidx=0;
  for (int ix=1; ix<ntypes; ++ix)
    for (int iy=ix; iy<ntypes; ++iy){
      host_write_int[ix*lj_types+iy] = host_write_int[iy*lj_types+ix] = tbidx; // tabindex
      tbidx+=n_table;
    }
  ucl_copy(tableindex,host_write_int,false);

  n_pair=ntypes*(ntypes-1)/2; //ntypes is larger by true n_type by one (added when calling init func.)
  // UCL_H_Vec<numtyp> host_write(n_pair*n_table,*(this->ucl_device),
                                //  UCL_WRITE_ONLY);
  // alloc table, copy is done in sendTable
  frc_tb.alloc(n_pair*n_table,*(this->ucl_device),UCL_READ_ONLY);

  UCL_H_Vec<double> dview;
  sp_lj.alloc(4,*(this->ucl_device),UCL_READ_ONLY);
  dview.view(host_special_lj,4,*(this->ucl_device));
  ucl_copy(sp_lj,dview,false);

  _allocated=true;
  this->_max_bytes=frc_tb.row_bytes()+tableindex.row_bytes()+sp_lj.row_bytes();
  return 0;
}

template <class numtyp, class acctyp>
void XNDAFT::clear() {
  if (!_allocated)
    return;
  _allocated=false;

  frc_tb.clear();
  tableindex.clear();
  sp_lj.clear();
  this->clear_atomic();
}

template <class numtyp, class acctyp>
double XNDAFT::host_memory_usage() const {
  return this->host_memory_usage_atomic()+sizeof(XNDAF<numtyp,acctyp>);
}

template <class numtyp, class acctyp>
void XNDAFT:: sendTable(double **tb) {
  UCL_H_Vec<numtyp> host_write(n_pair*n_table,*(this->ucl_device),
                                 UCL_WRITE_ONLY);
  int tbidx=0;
  for(int p=0;p<n_pair;++p)
    for(int i=0;i<n_table;++i)
      host_write[tbidx++]=tb[i][p];
  ucl_copy(frc_tb,host_write,false);
}

// ---------------------------------------------------------------------------
// Calculate energies, forces, and torques
// ---------------------------------------------------------------------------
template <class numtyp, class acctyp>
int XNDAFT::loop(const int eflag, const int vflag) {
  // Compute the block size and grid size to keep all cores busy
  const int BX=this->block_size();
  int GX=static_cast<int>(ceil(static_cast<double>(this->ans->inum())/
                               (BX/this->_threads_per_atom)));

  int ainum=this->ans->inum();
  int nbor_pitch=this->nbor->nbor_pitch();
  this->time_pair.start();
  if (shared_types) {
    this->k_pair_sel->set_size(GX,BX);
    this->k_pair_sel->run(&this->atom->x, &cutsq, &drinv, &n_table, &tableindex, &frc_tb,  &sp_lj,
                          &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                          &this->ans->force, &this->ans->engv, &eflag, &vflag,
                          &ainum, &nbor_pitch, &this->_threads_per_atom);
  } else {
    this->k_pair.set_size(GX,BX);
    this->k_pair.run(&this->atom->x, &cutsq, &drinv, &n_table, &tableindex, &frc_tb, &_lj_types, &sp_lj,
                     &this->nbor->dev_nbor, &this->_nbor_data->begin(),
                     &this->ans->force, &this->ans->engv, &eflag, &vflag,
                     &ainum, &nbor_pitch, &this->_threads_per_atom);
  }
  this->time_pair.stop();
  return GX;
}

template class XNDAF<PRECISION,ACC_PRECISION>;
}
