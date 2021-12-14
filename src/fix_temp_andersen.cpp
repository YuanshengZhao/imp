// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_temp_andersen.h"

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "modify.h"
#include "update.h"
#include "variable.h"
#include "domain.h"
#include "random_park.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum{CONSTANT,EQUAL};

/* ---------------------------------------------------------------------- */

FixTempAndersen::FixTempAndersen(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  tstr(nullptr), random(nullptr)
{
  if (narg < 8) error->all(FLERR,"Illegal fix temp/andersen command");

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix temp/andersen command");

  restart_global = 1;
  scalar_flag = 1;
  global_freq = nevery;
  extscalar = 1;
  ecouple_flag = 1;
  dynamic_group_allow = 1;

  tstr = nullptr;
  if (utils::strmatch(arg[4],"^v_")) {
    tstr = utils::strdup(arg[4]+2);
    tstyle = EQUAL;
  } else {
    t_start = utils::numeric(FLERR,arg[4],false,lmp);
    t_target = t_start;
    tstyle = CONSTANT;
  }

  t_stop = utils::numeric(FLERR,arg[5],false,lmp);
  fraction = utils::numeric(FLERR,arg[7],false,lmp);

  energy = 0.0;

  random = new RanPark(lmp,utils::numeric(FLERR,arg[6],false,lmp));
}

/* ---------------------------------------------------------------------- */

FixTempAndersen::~FixTempAndersen()
{
  delete [] tstr;
}

/* ---------------------------------------------------------------------- */

int FixTempAndersen::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTempAndersen::init()
{
  // check variable

  if (tstr) {
    tvar = input->variable->find(tstr);
    if (tvar < 0)
      error->all(FLERR,"Variable name for fix temp/andersen does not exist");
    if (input->variable->equalstyle(tvar)) tstyle = EQUAL;
    else error->all(FLERR,"Variable for fix temp/andersen is invalid style");
  }

}

/* ---------------------------------------------------------------------- */

void FixTempAndersen::end_of_step()
{
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  // set current t_target
  // if variable temp, evaluate variable, wrap with clear/add

  if (tstyle == CONSTANT)
    t_target = t_start + delta * (t_stop-t_start);
  else {
    modify->clearstep_compute();
    t_target = input->variable->compute_equal(tvar);
    if (t_target < 0.0)
      error->one(FLERR,
                 "Fix temp/andersen variable returned negative temperature");
    modify->addstep_compute(update->ntimestep + nevery);
  }

  // rescale velocity of appropriate atoms if outside window

  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int dim = domain->dimension;
  int *type = atom->type;
  double factor,pfactor=force->boltz/force->mvv2e;
  double *rmass = atom->rmass;
  double *mass = atom->mass;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if(random->uniform()>=fraction) continue;
      if (rmass) factor = sqrt(pfactor*t_target/rmass[i]);
      else factor = sqrt(pfactor*t_target/mass[type[i]]);
      v[i][0] = random->gaussian() * factor;
      v[i][1] = random->gaussian() * factor;
      if (dim == 3) v[i][2] = random->gaussian() * factor;
      else v[i][2] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixTempAndersen::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

double FixTempAndersen::compute_scalar()
{
  return energy;
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *FixTempAndersen::extract(const char *str, int &dim)
{
  if (strcmp(str,"t_target") == 0) {
    dim = 0;
    return &t_target;
  }
  return nullptr;
}
