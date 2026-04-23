#ifndef SRC_DRIFT_KINETIC_PARTICLES_H
#define SRC_DRIFT_KINETIC_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace drift_kinetic {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);
  PetscErrorCode initialize_point_by_field(const Arr B_arr);
  PetscErrorCode finalize() override;

  PetscErrorCode sync_dk_curr_storage();
  PetscErrorCode prepare_storage();
  PetscErrorCode form_iteration();

  PetscReal kinetic_energy_local() const;
  PetscReal get_average_iteration_number() const;
  PetscInt get_max_iteration_number() const;
  const std::vector<std::list<PointByField>>& get_dk_curr_storage() const
  {
    return dk_curr_storage;
  }

  Vec M;
  Vec M_loc;
  Arr M_arr;
  Arr B0_arr;

protected:
  PetscReal n_Np(const PointByField& point) const;
  PetscReal qn_Np(const PointByField& point) const;
  PetscErrorCode update_cells_seq();
  PetscErrorCode update_cells_mpi();
  PetscErrorCode correct_coordinates(PointByField& point);
  std::vector<std::list<PointByField>> dk_curr_storage;
  std::vector<std::vector<PointByField>> dk_prev_storage;
  PetscInt size = 0;
  PetscReal avgit = 0.0;
  PetscInt maxit = 0;
  Simulation& simulation_;
};

}  // namespace drift_kinetic

#endif  // SRC_DRIFT_KINETIC_PARTICLES_H
