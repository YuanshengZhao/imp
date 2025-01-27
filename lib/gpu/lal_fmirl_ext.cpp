/***************************************************************************
                                 fmirl_ext.cpp
                             -------------------
                            Trung Dac Nguyen (ORNL)

  Functions for LAMMPS access to fmirl acceleration routines.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : nguyentd@ornl.gov
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_fmirl.h"

using namespace std;
using namespace LAMMPS_AL;

static FMIRL<PRECISION,ACC_PRECISION> FMIRLMF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int fmirl_gpu_init(const int ntypes, const int ntable, double host_cutsq,
           double host_dr, double *host_special_lj,
           const int inum, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           int &gpu_mode, FILE *screen) {
  FMIRLMF.clear();
  gpu_mode=FMIRLMF.device->gpu_mode();
  double gpu_split=FMIRLMF.device->particle_split();
  int first_gpu=FMIRLMF.device->first_device();
  int last_gpu=FMIRLMF.device->last_device();
  int world_me=FMIRLMF.device->world_me();
  int gpu_rank=FMIRLMF.device->gpu_rank();
  int procs_per_gpu=FMIRLMF.device->procs_per_gpu();

  FMIRLMF.device->init_message(screen,"fmirl",first_gpu,last_gpu);

  bool message=false;
  if (FMIRLMF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=FMIRLMF.init(ntypes,ntable,host_cutsq,
           host_dr,host_special_lj,
           inum,nall,max_nbors,
           maxspecial,cell_size,
           gpu_split, screen);

  FMIRLMF.device->world_barrier();
  if (message)
    fprintf(screen,"Done.\n");

  for (int i=0; i<procs_per_gpu; i++) {
    if (message) {
      if (last_gpu-first_gpu==0)
        fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
      else
        fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                last_gpu,i);
      fflush(screen);
    }
    if (gpu_rank==i && world_me!=0)
      init_ok=FMIRLMF.init(ntypes,ntable,host_cutsq,
           host_dr,host_special_lj,
           inum,nall,max_nbors,
           maxspecial,cell_size,
           gpu_split, screen);

    FMIRLMF.device->gpu_barrier();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    FMIRLMF.estimate_gpu_overhead();
  return init_ok;
}

void fmirl_gpu_clear() {
  FMIRLMF.clear();
}
void fmirl_gpu_sendTable(double **host_u0,double **host_f0,double **host_uf,double **host_ff) {
  FMIRLMF.sendTable(host_u0,host_f0,host_uf,host_ff);
}

int ** fmirl_gpu_compute_n(const int ago, const int inum_full,
                           const int nall, double **host_x, int *host_type,
                           double *sublo, double *subhi, tagint *tag, int **nspecial,
                           tagint **special, const bool eflag, const bool vflag,
                           const bool eatom, const bool vatom, int &host_start,
                           int **ilist, int **jnum, const double cpu_time,
                           bool &success) {
  return FMIRLMF.compute(ago, inum_full, nall, host_x, host_type, sublo,
                      subhi, tag, nspecial, special, eflag, vflag, eatom,
                      vatom, host_start, ilist, jnum, cpu_time, success);
}

void fmirl_gpu_compute(const int ago, const int inum_full, const int nall,
                       double **host_x, int *host_type, int *ilist, int *numj,
                       int **firstneigh, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       const double cpu_time, bool &success) {
  FMIRLMF.compute(ago,inum_full,nall,host_x,host_type,ilist,numj,
               firstneigh,eflag,vflag,eatom,vatom,host_start,cpu_time,success);
}

double fmirl_gpu_bytes() {
  return FMIRLMF.host_memory_usage();
}


