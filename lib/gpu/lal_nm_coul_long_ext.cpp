/***************************************************************************
                            nm_coul_long_ext.cpp
                             -------------------
                            W. Michael Brown (ORNL)

  Functions for LAMMPS access to nm/cut/coul/long acceleration routines.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                :
    email                : brownw@ornl.gov
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_nm_coul_long.h"

using namespace std;
using namespace LAMMPS_AL;

static NMCoulLong<PRECISION,ACC_PRECISION> NMCLMF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int nmcl_gpu_init(const int ntypes, double **cutsq, double **host_gma,
                  double **host_fnm, double **host_nn, double **host_mm,
                  double **host_r02,
                  double **offset, double *special_lj, const int inum,
                  const int nall, const int max_nbors, const int maxspecial,
                  const double cell_size, int &gpu_mode, FILE *screen,
                  double **host_cut_ljsq, double host_cut_coulsq,
                  double *host_special_coul, const double qqrd2e,
                  const double g_ewald) {
  NMCLMF.clear();
  gpu_mode=NMCLMF.device->gpu_mode();
  double gpu_split=NMCLMF.device->particle_split();
  int first_gpu=NMCLMF.device->first_device();
  int last_gpu=NMCLMF.device->last_device();
  int world_me=NMCLMF.device->world_me();
  int gpu_rank=NMCLMF.device->gpu_rank();
  int procs_per_gpu=NMCLMF.device->procs_per_gpu();

  NMCLMF.device->init_message(screen,"nm/cut/coul/long",first_gpu,last_gpu);

  bool message=false;
  if (NMCLMF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=NMCLMF.init(ntypes, cutsq, host_gma, host_fnm, host_nn, host_mm, host_r02,
                        offset, special_lj, inum, nall, max_nbors, maxspecial,
                        cell_size, gpu_split, screen, host_cut_ljsq,
                        host_cut_coulsq, host_special_coul, qqrd2e, g_ewald);

  NMCLMF.device->world_barrier();
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
      init_ok=NMCLMF.init(ntypes, cutsq, host_gma, host_fnm, host_nn, host_mm, host_r02,
                          offset, special_lj, inum, nall, max_nbors, maxspecial,
                          cell_size, gpu_split, screen, host_cut_ljsq,
                          host_cut_coulsq, host_special_coul, qqrd2e, g_ewald);

    NMCLMF.device->gpu_barrier();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    NMCLMF.estimate_gpu_overhead();
  return init_ok;
}

// ---------------------------------------------------------------------------
// Copy updated coeffs from host to device
// ---------------------------------------------------------------------------
void nmcl_gpu_reinit(const int ntypes, double **cutsq, double **host_gma,
                    double **host_fnm, double **host_nn, double **host_mm,
                    double **host_r02,
                    double **offset, double **host_cut_ljsq) {
  int world_me=NMCLMF.device->world_me();
  int gpu_rank=NMCLMF.device->gpu_rank();
  int procs_per_gpu=NMCLMF.device->procs_per_gpu();

  if (world_me==0)
    NMCLMF.reinit(ntypes, cutsq, host_gma, host_fnm, host_nn, host_mm, host_r02,
                  offset, host_cut_ljsq);
  NMCLMF.device->world_barrier();

  for (int i=0; i<procs_per_gpu; i++) {
    if (gpu_rank==i && world_me!=0)
      NMCLMF.reinit(ntypes, cutsq, host_gma, host_fnm, host_nn, host_mm, host_r02,
                    offset, host_cut_ljsq);
    NMCLMF.device->gpu_barrier();
  }
}

void nmcl_gpu_clear() {
  NMCLMF.clear();
}

int** nmcl_gpu_compute_n(const int ago, const int inum_full,
                         const int nall, double **host_x, int *host_type,
                         double *sublo, double *subhi, tagint *tag, int **nspecial,
                         tagint **special, const bool eflag, const bool vflag,
                         const bool eatom, const bool vatom, int &host_start,
                         int **ilist, int **jnum,  const double cpu_time,
                         bool &success, double *host_q, double *boxlo,
                         double *prd) {
  return NMCLMF.compute(ago, inum_full, nall, host_x, host_type, sublo,
                        subhi, tag, nspecial, special, eflag, vflag, eatom,
                        vatom, host_start, ilist, jnum, cpu_time, success,
                        host_q, boxlo, prd);
}

void nmcl_gpu_compute(const int ago, const int inum_full, const int nall,
                      double **host_x, int *host_type, int *ilist, int *numj,
                      int **firstneigh, const bool eflag, const bool vflag,
                      const bool eatom, const bool vatom, int &host_start,
                      const double cpu_time, bool &success, double *host_q,
                      const int nlocal, double *boxlo, double *prd) {
  NMCLMF.compute(ago,inum_full,nall,host_x,host_type,ilist,numj,
                firstneigh,eflag,vflag,eatom,vatom,host_start,cpu_time,success,
                host_q,nlocal,boxlo,prd);
}

double nmcl_gpu_bytes() {
  return NMCLMF.host_memory_usage();
}


