#include "particles.h"
#include "src/algorithms/drift_kinetic_push.h"
#include "src/algorithms/drift_kinetic_implicit.h"
#include "src/impls/drift_kinetic/simulation.h"
#include "src/impls/eccapfim/cell_traversal.h"
#include "src/utils/geometries.h"
#include "src/utils/utils.h"

namespace drift_kinetic {
namespace {

constexpr PetscReal periodic_split_eps = 1e-12;

struct CutBounds {
  Vector3R lower;
  Vector3R upper;
};

std::vector<std::vector<Vector3R>> split_coords_by_periodic_bounds( //
  const std::vector<Vector3R>& coords,                              //
  const CutBounds& cut_bounds,                                       //
  const DMBoundaryType bounds[3])
{
  std::vector<std::vector<Vector3R>> chunks;
  if (coords.empty())
    return chunks;

  const Vector3R& cut_lower = cut_bounds.lower;
  const Vector3R& cut_upper = cut_bounds.upper;

  auto is_periodic = [&](Axis axis) {
    return bounds[axis] == DM_BOUNDARY_PERIODIC;
  };

  Vector3R shift{};
  chunks.push_back({coords.front()});

  for (PetscInt i = 1; i < (PetscInt)coords.size(); ++i) {
    const auto& raw = coords[i];
    Vector3R curr = raw + shift;
    Vector3R delta{};
    bool crossed = false;

    for (Axis axis : {X, Y, Z}) {
      if (!is_periodic(axis))
        continue;

      while (curr[axis] > cut_upper[axis] + periodic_split_eps ||
             curr[axis] < cut_lower[axis] - periodic_split_eps) {
        PointByField wrapped(curr, 0.0, 0.0, 0.0);
        const PetscReal before = wrapped.r[axis];
        g_bound_periodic(wrapped, axis);
        const PetscReal after = wrapped.r[axis];
        if (std::abs(after - before) <= periodic_split_eps)
          break;

        delta[axis] += after - before;
        curr[axis] = after;
        crossed = true;
      }
    }

    if (crossed) {
      const Vector3R prev_shifted = chunks.back().back() + delta;
      shift += delta;
      chunks.push_back({prev_shifted, raw + shift});
      continue;
    }

    chunks.back().push_back(curr);
  }

  return chunks;
}

}  // namespace

Particles::Particles(Simulation& simulation, const SortParameters& parameters)
  : interfaces::Particles(simulation.world, parameters),
    dk_curr_storage(world.size.elements_product()),
    dk_prev_storage(world.size.elements_product()),
    simulation_(simulation)
{
  PetscMPIInt size;
  PetscCallAbort(PETSC_COMM_WORLD, MPI_Comm_size(PETSC_COMM_WORLD, &size));
  update_cells = (size == 1) //
    ? std::bind(std::mem_fn(&Particles::update_cells_seq), this)
    : std::bind(std::mem_fn(&Particles::update_cells_mpi), this);


  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &J));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &M));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(da, &Mn));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &J_loc));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &M_loc));
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateLocalVector(da, &Mn_loc));
}

PetscErrorCode Particles::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&Mn));
  PetscCall(VecDestroy(&J_loc));
  PetscCall(VecDestroy(&M_loc));
  PetscCall(VecDestroy(&Mn_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Particles::initialize_point_by_field(const Arr B_arr)
{
  PetscFunctionBeginUser;
  const PetscReal qm = parameters.q / parameters.m;
  const PetscReal mp = parameters.m;
  DriftKineticEsirkepov esirkepov(nullptr, B_arr, nullptr, nullptr);
  PetscCall(esirkepov.set_bounds(world.gstart, world.gsize));

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto& cell = storage[g];
    if (cell.empty())
      continue;

    auto& dk_cell = dk_curr_storage[g];
    dk_cell.clear();

    for (const auto& point : cell) {
      Vector3R B_p{};
      PetscCall(esirkepov.interpolate_B(B_p, point.r));

      dk_cell.emplace_back(point, B_p, mp, qm);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Particles::kinetic_energy_local() const
{
  PetscReal w = 0.0;
  const PetscReal mpw = parameters.n / static_cast<PetscReal>(parameters.Np);
#pragma omp parallel for reduction(+ : w)
  for (auto&& cell : dk_curr_storage) {
    for (auto&& point : cell) {
      w += (POW2(point.p_parallel) + POW2(point.p_perp));
    }
  }
  return 0.5 * parameters.m * mpw * w;
}

PetscReal Particles::get_average_iteration_number() const
{
  return avgit;
}

PetscReal Particles::get_average_number_of_traversed_cells() const
{
  return avgcell;
}


PetscErrorCode Particles::form_iteration()
{
  PetscFunctionBeginUser;
  PetscCall(DMDAVecGetArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecGetArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMDAVecGetArrayWrite(da, Mn_loc, &Mn_arr));
  PetscCall(DMDAVecGetArrayRead(da, simulation_.dBdx_loc, &simulation_.dBdx_arr));
  PetscCall(DMDAVecGetArrayRead(da, simulation_.dBdy_loc, &simulation_.dBdy_arr));
  PetscCall(DMDAVecGetArrayRead(da, simulation_.dBdz_loc, &simulation_.dBdz_arr));

  avgit = 0.0;
  avgcell = 0.0;

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;

  const CutBounds cut_bounds{
    {(world.start[X] - 0.5) * dx, (world.start[Y] - 0.5) * dy, (world.start[Z] - 0.5) * dz},
    {(world.end[X] + 0.5) * dx, (world.end[Y] + 0.5) * dy, (world.end[Z] + 0.5) * dz},
  };

  static const PetscReal max = std::numeric_limits<double>::max();

  auto process_bound = [&](PetscReal vh, PetscReal x, Axis axis) {
    if (vh > 0)
      return (cut_bounds.upper[axis] - x) / vh;
    else if (vh < 0)
      return (cut_bounds.lower[axis] - x) / vh;
    else
      return max;
  };

  DriftKineticEsirkepov util(E_arr, B_arr, J_arr, M_arr);
  DriftKineticEsirkepov util_temp(nullptr, B_arr, nullptr, Mn_arr);
  util.set_dBidrj_precomputed(simulation_.dBdx_arr, simulation_.dBdy_arr, simulation_.dBdz_arr);
  //PetscCall(util.set_bounds(world.gstart, world.gsize));
  const PetscReal inv_size = size > 0 ? 1.0 / static_cast<PetscReal>(size) : 0.0;

#pragma omp parallel for reduction(+ : VgradB, avgit, avgcell)
  for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
    const auto& prev_cell = dk_prev_storage[g];

    PetscInt i = 0;
    for (auto& curr : dk_curr_storage[g]) {
      auto prev(prev_cell[i]);

      /// @todo this part should reuse the logic from:
      /// tests/drift_kinetic_push/drift_kinetic_push.h:620 implicit_test_utils::interpolation_test()
      DriftKineticPush push(q / m, m);
      auto call_abort = [&](PetscErrorCode ierr) {
        if (ierr)
          PetscCallAbort(PETSC_COMM_WORLD, ierr);
      };
      push.set_fields_callback(
        [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p,
          Vector3R& gradB_p) {
          E_p = {};
          B_p = {};
          gradB_p = {};
          Vector3R Es_p, Bs_p, gradBs_p;

          Vector3R pos = (rn - r0);
          auto coords = cell_traversal(rn, r0);
          auto chunked_coords = split_coords_by_periodic_bounds(coords, cut_bounds, world.bounds);

          PetscReal path = 0.0;
          PetscInt segments = 0;
          pos = {};

          for (const auto& chunk : chunked_coords) {
            for (PetscInt s = 1; s < (PetscInt)chunk.size(); ++s) {
              auto ds = chunk[s] - chunk[s - 1];
              path += ds.length();
              pos += ds;
              ++segments;
            }
          }

          if (segments <= 0) {
            segments = 1;
          }

          pos[X] = pos[X] != 0 ? pos[X] / dx : (PetscReal)segments;
          pos[Y] = pos[Y] != 0 ? pos[Y] / dy : (PetscReal)segments;
          pos[Z] = pos[Z] != 0 ? pos[Z] / dz : (PetscReal)segments;

          for (const auto& chunk : chunked_coords) {
            for (PetscInt s = 1; s < (PetscInt)chunk.size(); ++s) {
              auto&& rs0 = chunk[s - 1];
              auto&& rsn = chunk[s - 0];
              call_abort(util.interpolate(Es_p, Bs_p, gradBs_p, rsn, rs0));

              PetscReal beta = path > 0 ? (rsn - rs0).length() / path : 1.0;
              E_p += Es_p * beta;
              B_p += Bs_p * beta;
              gradB_p += gradBs_p.elementwise_division(pos);
            }
          }
        });
      
      //PetscReal v = std::sqrt((POW2(curr.p_parallel) + POW2(curr.p_perp)));
      Vector3R Vph;
      PetscReal dtau = dt;
#if 0
      LOG("v_th = {}", v);
      LOG("v_drift = ({}, {}, {})", Vph.x(), Vph.y(), Vph.z());
#endif

      //for (PetscReal dtau = 0.0, tau = 0.0; tau < dt; tau += dtau) {
      //  PetscReal dtx = process_bound(Vph.x(), curr.x(), X);
      //  PetscReal dty = process_bound(Vph.y(), curr.y(), Y);
      //  PetscReal dtz = process_bound(Vph.z(), curr.z(), Z);
//
      //  dtau = std::min({dt - tau, dtx, dty, dtz});

        //LOG("dt - tau, dtx, dty, dtz = {}, {}, {}, {}", dt - tau, dtx, dty, dtz);

        push.process(dtau, curr, prev);
        avgit += (PetscReal)(push.get_iteration_number() + 1) * inv_size;
        Vph = (curr.r - prev.r) / dtau;

        //LOG("Vph.length(), v = {}, {}", Vph.length(), v);

        auto coords = cell_traversal(curr.r, prev.r);
        auto chunked_coords = split_coords_by_periodic_bounds(coords, cut_bounds, world.bounds);

        PetscReal path = 0.0;
        PetscInt segments = 0;
        for (const auto& chunk : chunked_coords) {
          segments += std::max<PetscInt>(0, (PetscInt)chunk.size() - 1);
          for (PetscInt s = 1; s < (PetscInt)chunk.size(); ++s)
            path += (chunk[s] - chunk[s - 1]).length();
        }
        avgcell += (PetscReal)segments * inv_size;

        PetscReal a0 = qn_Np(curr);
        PetscReal b0 = curr.mu_p * n_Np(curr);

        for (const auto& chunk : chunked_coords) {
          for (PetscInt s = 1; s < (PetscInt)chunk.size(); ++s) {
            auto&& rs0 = chunk[s - 1];
            auto&& rsn = chunk[s - 0];
            PetscReal a = a0 * (rsn - rs0).length() / path;
            call_abort(util.decomposition_J(rsn, rs0, Vph, a));

            PetscReal b = b0 * (rsn - rs0).length() / path;
            call_abort(util.decomposition_M(rsn, b));
            call_abort(util_temp.decomposition_M(rs0, b));
          }
        }
        //correct_coordinates(curr);
        //prev = curr;
        VgradB += push.get_VgradB(curr);
      ++i;
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(da, J_loc, &J_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, M_loc, &M_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, Mn_loc, &Mn_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, simulation_.dBdx_loc, &simulation_.dBdx_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, simulation_.dBdy_loc, &simulation_.dBdy_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, simulation_.dBdz_loc, &simulation_.dBdz_arr));
  PetscCall(DMLocalToGlobal(da, J_loc, ADD_VALUES, J));
  PetscCall(DMLocalToGlobal(da, M_loc, ADD_VALUES, M));
  PetscCall(DMLocalToGlobal(da, Mn_loc, ADD_VALUES, Mn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::after_iteration() {
  PetscFunctionBeginUser;

  PetscReal q = parameters.q;
  PetscReal m = parameters.m;
  const CutBounds cut_bounds{
    {(world.start[X] - 0.5) * dx, (world.start[Y] - 0.5) * dy, (world.start[Z] - 0.5) * dz},
    {(world.end[X] + 0.5) * dx, (world.end[Y] + 0.5) * dy, (world.end[Z] + 0.5) * dz},
  };

  DriftKineticEsirkepov util(nullptr, B_arr, nullptr, nullptr);

#pragma omp parallel for
  for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
    const auto& prev_cell = dk_prev_storage[g];

    PetscInt i = 0;
    for (auto& curr : dk_curr_storage[g]) {
      auto prev(prev_cell[i]);

      DriftKineticPush push(q / m, m);

      push.set_fields_callback(
        [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p, Vector3R& B_p,
          Vector3R& gradB_p) {
          E_p = {};
          B_p = {};
          gradB_p = {};
          Vector3R Bs_p;

          auto coords = cell_traversal(rn, r0);
          auto chunked_coords = split_coords_by_periodic_bounds(coords, cut_bounds, world.bounds);

          PetscReal path = 0.0;
          for (const auto& chunk : chunked_coords) {
            for (PetscInt s = 1; s < (PetscInt)chunk.size(); ++s)
              path += (chunk[s] - chunk[s - 1]).length();
          }

          for (const auto& chunk : chunked_coords) {
            for (PetscInt s = 1; s < (PetscInt)chunk.size(); ++s) {
              auto&& rs0 = chunk[s - 1];
              auto&& rsn = chunk[s - 0];
              util.interpolate_B(Bs_p, rsn);

              PetscReal beta = path > 0 ? (rsn - rs0).length() / path : 1.0;
              B_p += Bs_p * beta;
            }
          }
        });

      push.update_v_perp(curr, prev);
      i++;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscReal Particles::n_Np(const PointByField& point) const
{
  Point dummy(point.r, Vector3R{});
  return interfaces::Particles::n_Np(dummy);
}

PetscReal Particles::qn_Np(const PointByField& point) const
{
  Point dummy(point.r, Vector3R{});
  return interfaces::Particles::qn_Np(dummy);
}

PetscErrorCode Particles::sync_dk_curr_storage()
{
  PetscFunctionBeginUser;
  const PetscReal qm = parameters.q / parameters.m;
  const PetscReal mp = parameters.m;

  PetscCall(DMGlobalToLocal(simulation_.da, simulation_.B, INSERT_VALUES, simulation_.B_loc));
  PetscCall(DMDAVecGetArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));

  DriftKineticEsirkepov esirkepov(nullptr, simulation_.B_arr, nullptr, nullptr);

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto& cell = storage[g];
    if (cell.empty())
      continue;

    auto& dk_cell = dk_curr_storage[g];
    for (const auto& point : cell) {
      Vector3R B_p{};
      PetscCall(esirkepov.interpolate_B(B_p, point.r));
      PointByField point_by_field(point, B_p, mp, qm);
      dk_cell.emplace_back(point_by_field);
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(simulation_.da, simulation_.B_loc, &simulation_.B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::prepare_storage()
{
  PetscFunctionBeginUser;
  size = 0;
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    if (auto& curr = dk_curr_storage[g]; !curr.empty()) {
      auto& prev = dk_prev_storage[g];
      prev = std::vector(curr.begin(), curr.end());
      size += (PetscInt)curr.size();
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::correct_coordinates(PointByField& point)
{
  PetscFunctionBeginUser;
  if (world.bounds[X] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, X);
  if (world.bounds[Y] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, Y);
  if (world.bounds[Z] == DM_BOUNDARY_PERIODIC)
    g_bound_periodic(point, Z);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::update_cells_seq()
{
  PetscFunctionBeginUser;
  PetscLogEventBegin(events[0], 0, 0, 0, 0);

  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    auto it = dk_curr_storage[g].begin();
    while (it != dk_curr_storage[g].end()) {
      PetscCall(correct_coordinates(*it));

      Vector3I vng{
        FLOOR_STEP(it->x(), dx),
        FLOOR_STEP(it->y(), dy),
        FLOOR_STEP(it->z(), dz),
      };

      auto ng = world.s_g(REP3_A(vng));
      if (ng == g) {
        it = std::next(it);
        continue;
      }

      if (is_point_within_bounds(vng, world.start, world.size))
        dk_curr_storage[ng].emplace_back(std::move(*it));

      it = dk_curr_storage[g].erase(it);
    }
  }

  PetscLogEventEnd(events[0], 0, 0, 0, 0);

  PetscInt sum = 0;
  for (const auto& cell : dk_curr_storage)
    sum += cell.size();

  LOG("  Cells have been updated, total number of particles: {}", sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Particles::update_cells_mpi()
{
  PetscFunctionBeginUser;
  constexpr PetscInt neighbors_num = POW3(3);
  constexpr PetscInt center_index = indexing::petsc_index(1, 1, 1, 0, 3, 3, 3, 1);

  auto get_index = [](const Vector3I& r, Axis axis, const World& world) {
    if (r[axis] < world.start[axis])
      return 0;
    if (r[axis] < world.end[axis])
      return 1;
    return 2;
  };

  auto get_neighbor = [](PetscInt i, const World& world) {
    return world.neighbors[i] < 0 ? MPI_PROC_NULL : world.neighbors[i];
  };

  std::vector<PointByField> outgoing[neighbors_num];
  std::vector<PointByField> incoming[neighbors_num];

  PetscLogEventBegin(events[0], 0, 0, 0, 0);

  LOG("  Starting MPI cells update for \"{}\"", parameters.sort_name);
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    Vector3I pg{
      world.start[X] + g % world.size[X],
      world.start[Y] + (g / world.size[X]) % world.size[Y],
      world.start[Z] + (g / world.size[X]) / world.size[Y],
    };

    auto it = dk_curr_storage[g].begin();
    while (it != dk_curr_storage[g].end()) {
      Vector3I ng{
        FLOOR_STEP(it->x(), dx),
        FLOOR_STEP(it->y(), dy),
        FLOOR_STEP(it->z(), dz),
      };

      if (pg[X] == ng[X] && pg[Y] == ng[Y] && pg[Z] == ng[Z]) {
        it = std::next(it);
        continue;
      }

      PetscInt i = indexing::petsc_index( //
        get_index(ng, X, world),           //
        get_index(ng, Y, world),           //
        get_index(ng, Z, world),           //
        0, 3, 3, 3, 1);

      if (i == center_index) {
        PetscInt j = world.s_g(   //
          ng[X] - world.start[X], //
          ng[Y] - world.start[Y], //
          ng[Z] - world.start[Z]);

        dk_curr_storage[j].emplace_back(std::move(*it));
        it = dk_curr_storage[g].erase(it);
        continue;
      }

      PetscCall(correct_coordinates(*it));

      outgoing[i].emplace_back(std::move(*it));
      it = dk_curr_storage[g].erase(it);
    }
  }

  size_t o_num[neighbors_num];
  size_t i_num[neighbors_num];
  for (PetscInt i = 0; i < neighbors_num; ++i) {
    o_num[i] = outgoing[i].size();
    i_num[i] = 0;
  }

  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscInt req = 0;
  MPI_Request reqs[2 * (neighbors_num - 1)];

  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;

    PetscInt r = (neighbors_num - 1) - s;
    PetscCallMPI(MPI_Isend(&o_num[s], 1, MPIU_SIZE_T, get_neighbor(s, world), MPI_TAG_NUMBERS, comm, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(&i_num[r], 1, MPIU_SIZE_T, get_neighbor(r, world), MPI_TAG_NUMBERS, comm, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  req = 0;
  for (PetscInt s = 0; s < neighbors_num; ++s) {
    if (s == center_index)
      continue;

    PetscInt r = (neighbors_num - 1) - s;
    incoming[r].resize(i_num[r]);
    PetscCallMPI(MPI_Isend(outgoing[s].data(), o_num[s] * sizeof(PointByField), MPI_BYTE,
      get_neighbor(s, world), MPI_TAG_POINTS, comm, &reqs[req++]));
    PetscCallMPI(MPI_Irecv(incoming[r].data(), i_num[r] * sizeof(PointByField), MPI_BYTE,
      get_neighbor(r, world), MPI_TAG_POINTS, comm, &reqs[req++]));
  }
  PetscCallMPI(MPI_Waitall(req, reqs, MPI_STATUSES_IGNORE));

  for (PetscInt i = 0; i < neighbors_num; ++i) {
    if (i == center_index || i_num[i] == 0)
      continue;

    for (auto&& point : incoming[i]) {
      PetscInt g = world.s_g(  //
        FLOOR_STEP(point.x(), dx) - world.start[X],  //
        FLOOR_STEP(point.y(), dy) - world.start[Y],  //
        FLOOR_STEP(point.z(), dz) - world.start[Z]);

      dk_curr_storage[g].emplace_back(std::move(point));
    }
  }

  PetscLogEventEnd(events[0], 0, 0, 0, 0);

  const std::vector<std::pair<std::string, size_t*>> map{
    {"    sent particles ", o_num},
    {"    received particles ", i_num},
  };

  for (auto&& [op, num] : map) {
    PetscInt sum = 0;

    for (PetscInt i = 0; i < neighbors_num; ++i)
      sum += num[i];

    PetscCall(MPIUtils::log_statistics(op, sum, comm));
  }

  PetscInt sum = 0;
  for (const auto& cell : dk_curr_storage)
    sum += cell.size();

  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPIU_INT, MPI_SUM, comm));
  LOG("  Cells have been updated, total number of particles: {}", sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}


}  // namespace drift_kinetic
