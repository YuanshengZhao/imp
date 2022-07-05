// **************************************************************************
//                                   xndaf.cu
//                             -------------------
//                           Trung Dac Nguyen (ORNL)
//
//  Device code for acceleration of the xndaf pair style
//
// __________________________________________________________________________
//    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
// __________________________________________________________________________
//
//    begin                :
//    email                : nguyentd@ornl.gov
// ***************************************************************************

#if defined(NV_KERNEL) || defined(USE_HIP)
#include "lal_aux_fun1.h"
#ifndef _DOUBLE_DOUBLE
_texture( pos_tex,float4);
#else
_texture_2d( pos_tex,int4);
#endif
#else
#define pos_tex x_
#endif

__kernel void k_xndaf(const __global numtyp4 *restrict x_,
                    const numtyp cutsq,
                    const numtyp drinv,
                    const int n_table,
                    const __global int *restrict tabindex,
                    const __global numtyp *restrict frc_tb,
                    const int lj_types,
                    const __global numtyp *restrict sp_lj_in,
                    const __global int *dev_nbor,
                    const __global int *dev_packed,
                    __global acctyp4 *restrict ans,
                    __global acctyp *restrict engv,
                    const int eflag, const int vflag, const int inum,
                    const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp sp_lj[4];
  int n_stride;
  local_allocate_store_pair();

  sp_lj[0]=sp_lj_in[0];
  sp_lj[1]=sp_lj_in[1];
  sp_lj[2]=sp_lj_in[2];
  sp_lj[3]=sp_lj_in[3];

  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }

  if (ii<inum) {
    int nbor, nbor_end;
    int i, numj;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    int itype=ix.w;

    numtyp factor_lj;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      int jtype=jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;

      int mtype=itype*lj_types+jtype;
      if (rsq<cutsq) {
        rsq=ucl_sqrt(rsq);
        int tbi=(int)(rsq*drinv);
        numtyp force = (numtyp)0;
        if(tbi<n_table) {
          force=frc_tb[tbi+tabindex[mtype]]*factor_lj;
          f.x+=delx*force;
          f.y+=dely*force;
          f.z+=delz*force;
        }
      }

    } // for nbor
  } // if ii
  store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                ans,engv);
}

__kernel void k_xndaf_fast(const __global numtyp4 *restrict x_,
                         const numtyp cutsq,
                         const numtyp drinv,
                         const int n_table,
                         const __global int *restrict tabindex,
                         const __global numtyp *restrict frc_tb,
                         const __global numtyp *restrict sp_lj_in,
                         const __global int *dev_nbor,
                         const __global int *dev_packed,
                         __global acctyp4 *restrict ans,
                         __global acctyp *restrict engv,
                         const int eflag, const int vflag, const int inum,
                         const int nbor_pitch, const int t_per_atom) {
  int tid, ii, offset;
  atom_info(t_per_atom,ii,tid,offset);

  __local numtyp sp_lj[4];
  int n_stride;
  local_allocate_store_pair();

  if (tid<4)
    sp_lj[tid]=sp_lj_in[tid];

  acctyp4 f;
  f.x=(acctyp)0; f.y=(acctyp)0; f.z=(acctyp)0;
  acctyp energy, virial[6];
  if (EVFLAG) {
    energy=(acctyp)0;
    for (int i=0; i<6; i++) virial[i]=(acctyp)0;
  }

  __syncthreads();

  if (ii<inum) {
    int nbor, nbor_end;
    int i, numj;
    nbor_info(dev_nbor,dev_packed,nbor_pitch,t_per_atom,ii,offset,i,numj,
              n_stride,nbor_end,nbor);

    numtyp4 ix; fetch4(ix,i,pos_tex); //x_[i];
    int iw=ix.w;
    int itype=fast_mul((int)MAX_SHARED_TYPES,iw);

    numtyp factor_lj;
    for ( ; nbor<nbor_end; nbor+=n_stride) {

      int j=dev_packed[nbor];
      factor_lj = sp_lj[sbmask(j)];
      j &= NEIGHMASK;

      numtyp4 jx; fetch4(jx,j,pos_tex); //x_[j];
      int mtype=itype+jx.w;

      // Compute r12
      numtyp delx = ix.x-jx.x;
      numtyp dely = ix.y-jx.y;
      numtyp delz = ix.z-jx.z;
      numtyp rsq = delx*delx+dely*dely+delz*delz;

      if (rsq<cutsq) {
        rsq=ucl_sqrt(rsq);
        int tbi=(int)(rsq*drinv);
        numtyp force = (numtyp)0;
        if(tbi<n_table) {
          force=frc_tb[tbi+tabindex[mtype]]*factor_lj;
          f.x+=delx*force;
          f.y+=dely*force;
          f.z+=delz*force;
        }
      }

    } // for nbor
  } // if ii
  store_answers(f,energy,virial,ii,inum,tid,t_per_atom,offset,eflag,vflag,
                ans,engv);
}
