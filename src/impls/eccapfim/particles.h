#ifndef SRC_ECCAPFIM_PARTICLES_H
#define SRC_ECCAPFIM_PARTICLES_H

#include "src/pch.h"
#include "src/interfaces/particles.h"
#include "src/utils/shape.h"

namespace eccapfim {

class Simulation;

class Particles : public interfaces::Particles {
public:
  Particles(Simulation& simulation, const SortParameters& parameters);

  PetscErrorCode clear_sources();
  PetscErrorCode prepare_storage();
  PetscErrorCode form_iteration();

  PetscReal get_average_iteration_number() const;
  PetscReal get_average_number_of_traversed_cells() const;
  PetscInt get_maximum_number_of_traversed_cells() const;

protected:
  static constexpr const auto& shape_func = spline_of_2nd_order;
  static constexpr const auto& shape_radius = 1.5;

  /// @note We should iterate the `Point` ~ (x^{n+1,k}, v^{n+1,k}) from _previous_
  /// timestep, meaning that we have to store copy of `Particles::storage`.
  std::vector<std::vector<Point>> previous_storage;

  PetscInt size = 0;
  PetscReal avgit = 0;
  PetscReal avgcell = 0;
  PetscInt  maxcell = 0;

  Simulation& simulation_;
};


/**
 * @brief Finds a stopping points at cell edges along the
 * straight line between `start` and `end` points. On Yee
 * grid, edges, where electric field components are defined,
 * are shifted by a halfstep from nodes, obtained by `std::floor()`.
 *
 * @param[in] end   Last position of the particle.
 * @param[in] start Initial position of the particle.
 * @returns The sequence of start, intermediate stopping-points and the end point.
 *
 * @note The original implementation is taken from the repositories
 * https://github.com/francisengelmann/fast_voxel_traversal,
 * https://github.com/cgyurgyik/fast-voxel-traversal-algorithm.
 *
 * For a detailed explanation of the method see
 * Amanatides, John, and Andrew Woo. "A fast voxel traversal algorithm for ray tracing." Eurographics. Vol. 87. No. 3. 1987.
 */
std::vector<Vector3R> cell_traversal(const Vector3R& end, const Vector3R& start);

}  // namespace eccapfim

#endif  // SRC_ECCAPFIM_PARTICLES_H
