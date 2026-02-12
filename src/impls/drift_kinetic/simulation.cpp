#include "simulation.h"

#include "src/algorithms/drift_kinetic_implicit.h"
#include "src/utils/geometries.h"
#include "src/utils/operators.h"
#include "src/utils/utils.h"


namespace drift_kinetic {

static constexpr PetscReal atol = 1e-7;
static constexpr PetscReal rtol = 1e-7;
static constexpr PetscReal stol = 1e-7;
static constexpr PetscReal divtol = PETSC_DETERMINE;
static constexpr PetscInt maxit = 100;
static constexpr PetscInt maxf = PETSC_UNLIMITED;

namespace {

struct ChargeShape {
  static constexpr PetscReal shr = 1.5;
  static constexpr PetscInt shw = (PetscInt)(2.0 * shr);
  static constexpr PetscInt shm = POW3(shw);
  static constexpr PetscReal (&sfunc)(PetscReal) = spline_of_2nd_order;

  Vector3I start;
  PetscReal cache[shm];

  void setup(const Vector3R& r)
  {
    Vector3R p_r = Shape::make_r(r);

    start = Vector3I{
      (PetscInt)(std::ceil(p_r[X] - shr)),
      (PetscInt)(std::ceil(p_r[Y] - shr)),
      (PetscInt)(std::ceil(p_r[Z] - shr)),
    };

#pragma omp simd
    for (PetscInt i = 0; i < shm; ++i) {
      const auto g_x = (PetscReal)(start[X] + i % shw);
      const auto g_y = (PetscReal)(start[Y] + (i / shw) % shw);
      const auto g_z = (PetscReal)(start[Z] + (i / shw) / shw);
      cache[i] = sfunc(p_r[X] - g_x) * sfunc(p_r[Y] - g_y) * sfunc(p_r[Z] - g_z);
    }
  }
};

}  // namespace

PetscErrorCode Simulation::initialize_implementation()
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(da, &E));
  PetscCall(DMCreateGlobalVector(da, &B));
  PetscCall(DMCreateGlobalVector(da, &B0));
  PetscCall(DMCreateGlobalVector(da, &J));
  PetscCall(DMCreateGlobalVector(da, &M));
  PetscCall(DMCreateGlobalVector(da, &Mn));
  PetscCall(DMCreateGlobalVector(da, &dBdx));
  PetscCall(DMCreateGlobalVector(da, &dBdy));
  PetscCall(DMCreateGlobalVector(da, &dBdz));
  PetscCall(DMCreateGlobalVector(da, &E_hk));
  PetscCall(DMCreateGlobalVector(da, &B_hk));
  PetscCall(DMCreateLocalVector(da, &E_loc));
  PetscCall(DMCreateLocalVector(da, &B_loc));
  PetscCall(DMCreateLocalVector(da, &dBdx_loc));
  PetscCall(DMCreateLocalVector(da, &dBdy_loc));
  PetscCall(DMCreateLocalVector(da, &dBdz_loc));

  PetscCall(DMSetMatrixPreallocateOnly(da, PETSC_FALSE));
  PetscCall(DMSetMatrixPreallocateSkip(da, PETSC_TRUE));

  Rotor rotor(da);
  PetscCall(rotor.create_positive(&rotE));
  PetscCall(rotor.create_negative(&rotM));
  PetscCall(rotor.create_negative(&rotB));
  PetscCall(MatScale(rotB, -1)); /// @see `Simulation::form_function()`

  PetscInt gn[3];
  PetscInt procs[3];
  PetscInt s;
  DMBoundaryType bounds[3];
  DMDAStencilType st;
  PetscCall(DMDAGetInfo(da, NULL, REP3_A(&gn), REP3_A(&procs), NULL, &s, REP3_A(&bounds), &st));

  const PetscInt* lgn[3];
  PetscCall(DMDAGetOwnershipRanges(da, REP3_A(&lgn)));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), st, REP3_A(gn), REP3_A(procs), 6, st, REP3_A(lgn), &da_EB));
  PetscCall(DMSetUp(da_EB));

  PetscCall(DMCreateGlobalVector(da_EB, &sol));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, SNESNGMRES));
  PetscCall(SNESSetTolerances(snes, atol, rtol, stol, maxit, maxf));
  PetscCall(SNESSetDivergenceTolerance(snes, divtol));
  PetscCall(SNESSetFunction(snes, NULL, Simulation::form_iteration, this));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(init_particles(*this, particles_));

  energy_cons = std::make_unique<EnergyConservation>(*this);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(energy_cons->diagnose(geom_nt));
  PetscCall(energy_cons->finalize());
  PetscCall(interfaces::Simulation::finalize());

  PetscCall(VecDestroy(&E));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&B0));
  PetscCall(VecDestroy(&J));
  PetscCall(VecDestroy(&M));
  PetscCall(VecDestroy(&Mn));
  PetscCall(VecDestroy(&dBdx));
  PetscCall(VecDestroy(&dBdy));
  PetscCall(VecDestroy(&dBdz));
  PetscCall(VecDestroy(&E_hk));
  PetscCall(VecDestroy(&B_hk));
  PetscCall(VecDestroy(&E_loc));
  PetscCall(VecDestroy(&B_loc));
  PetscCall(VecDestroy(&dBdx_loc));
  PetscCall(VecDestroy(&dBdy_loc));
  PetscCall(VecDestroy(&dBdz_loc));

  PetscCall(MatDestroy(&rotE));
  PetscCall(MatDestroy(&rotB));
  PetscCall(MatDestroy(&rotM));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&sol));
  PetscCall(DMDestroy(&da_EB));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::timestep_implementation(PetscInt t)
{
  PetscFunctionBeginUser;
  for (auto& sort : particles_)
    PetscCall(sort->prepare_storage());
  PetscCall(energy_cons->diagnose(t - 1));

  LOG("to_snes():");
  /// @note Solution is initialized with guess before it is passed into `SNESSolve()`.
  /// The simplest choice is: (E^{n+1/2, k=0}, B^{n+1/2, k=0}) = (E^{n}, B^{n}).
  PetscCall(to_snes(E, B, sol));
  LOG("to_snes() has finished, SNESSolve():");
  PetscCall(SNESSolve(snes, NULL, sol));
  PetscCall(SNESGetIterationNumber(snes, &last_field_itnum));

  LOG("SNESSolve() has finished, SNESConvergedReasonView():");
  PetscCall(SNESConvergedReasonView(snes, PETSC_VIEWER_STDOUT_WORLD));

  // SNESConvergedReason reason;
  // PetscCall(SNESGetConvergedReason(snes, &reason));
  // PetscCheck(reason >= 0, PetscObjectComm((PetscObject)snes), PETSC_ERR_NOT_CONVERGED, "SNESSolve has not converged");

  PetscCall(SNESGetSolution(snes, &sol));
  PetscCall(from_snes(sol, E_hk, B_hk));
  PetscCall(VecAXPBY(E, 2, -1, E_hk));
  PetscCall(VecAXPBY(B, 2, -1, B_hk));

  for (auto& sort : particles_) {

    LOG("after_iteration() start:");
    PetscCall(DMGlobalToLocal(da, E, INSERT_VALUES, E_loc));
    PetscCall(DMGlobalToLocal(da, B, INSERT_VALUES, B_loc));
    PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
    PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    PetscCall(sort->after_iteration());

    PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
    PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));

    LOG("after_iteration() end:");
    PetscCall(sort->update_cells());
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_iteration(
  SNES /* snes */, Vec vx, Vec vf, void* ctx)
{
  PetscFunctionBeginUser;
  auto* simulation = (Simulation*)ctx;
  PetscCall(simulation->from_snes(vx, simulation->E_hk, simulation->B_hk));
  PetscCall(simulation->prepare_dBdr());
  PetscCall(simulation->form_current());
  PetscCall(simulation->form_function(vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::prepare_dBdr()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(dBdx_loc, 0));
  PetscCall(VecSet(dBdy_loc, 0));
  PetscCall(VecSet(dBdz_loc, 0));
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));
  PetscCall(DMDAVecGetArrayWrite(da, dBdx_loc, &dBdx_arr));
  PetscCall(DMDAVecGetArrayWrite(da, dBdy_loc, &dBdy_arr));
  PetscCall(DMDAVecGetArrayWrite(da, dBdz_loc, &dBdz_arr));

  DriftKineticEsirkepov dk_util(nullptr, B_arr, nullptr, nullptr);
  PetscCall(dk_util.set_dBidrj_local(dBdx_arr, dBdy_arr, dBdz_arr, world.start, world.size));

  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, dBdx_loc, &dBdx_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, dBdy_loc, &dBdy_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, dBdz_loc, &dBdz_arr));
  PetscCall(DMLocalToGlobal(da, dBdx_loc, ADD_VALUES, dBdx));
  PetscCall(DMLocalToGlobal(da, dBdy_loc, ADD_VALUES, dBdy));
  PetscCall(DMLocalToGlobal(da, dBdz_loc, ADD_VALUES, dBdz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_current()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(J, 0.0));
  PetscCall(VecSet(M, 0.0));
  PetscCall(VecSet(Mn, 0.0));
  VgradB = 0.;

  for (auto& sort : particles_) {
    PetscCall(VecSet(sort->J, 0.0));
    PetscCall(VecSet(sort->M, 0.0));
    PetscCall(VecSet(sort->Mn, 0.0));
    PetscCall(VecSet(sort->J_loc, 0.0));
    PetscCall(VecSet(sort->M_loc, 0.0));
    PetscCall(VecSet(sort->Mn_loc, 0.0));
    sort->VgradB = 0.;
  }

  PetscCall(DMGlobalToLocal(da, E_hk, INSERT_VALUES, E_loc));
  PetscCall(DMGlobalToLocal(da, B_hk, INSERT_VALUES, B_loc));
  PetscCall(DMDAVecGetArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, B_loc, &B_arr));

  for (auto& sort : particles_) {
    sort->E_arr = E_arr;
    sort->B_arr = B_arr;
    PetscCall(sort->form_iteration());
    PetscCall(VecAXPY(J, 1, sort->J));
    PetscCall(VecAXPY(M, 1, sort->M));
    PetscCall(VecAXPY(Mn, 1, sort->Mn));
    VgradB += sort->VgradB;
  }

  PetscCall(DMDAVecRestoreArrayRead(da, E_loc, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_loc, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::form_function(Vec vf)
{
  PetscFunctionBeginUser;
  Vec E_f, B_f;
  PetscCall(DMGetGlobalVector(da, &E_f));
  PetscCall(DMGetGlobalVector(da, &B_f));

  // F(E) = (E^{n+1/2,k} - E^{n}) / (dt / 2) + J^{n+1/2,k} - rot(B^{n+1/2,k}) + rot(M^{n+1/2,k}})
  PetscCall(VecAXPBYPCZ(E_f, +2 / dt, -2 / dt, 0, E_hk, E));
  PetscCall(VecAXPY(E_f, +1, J));
  PetscCall(MatMultAdd(rotM, M, E_f, E_f));
  PetscCall(MatMultAdd(rotB, B_hk, E_f, E_f));

  // F(B) = (B^{n+1/2,k} - B^{n}) / (dt / 2) + rot(E^{n+1/2,k})
  PetscCall(VecAXPBYPCZ(B_f, +2 / dt, -2 / dt, 0, B_hk, B));
  PetscCall(MatMultAdd(rotE, E_hk, B_f, B_f));

  PetscCall(to_snes(E_f, B_f, vf));

  PetscCall(DMRestoreGlobalVector(da, &E_f));
  PetscCall(DMRestoreGlobalVector(da, &B_f));
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Simulation::from_snes(Vec v, Vec vE, Vec vB)
{
  PetscFunctionBeginUser;
  const PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayWrite(da, vE, &E_arr));
  PetscCall(DMDAVecGetArrayWrite(da, vB, &B_arr));
  PetscCall(DMDAVecGetArrayDOFRead(da_EB, v, &arr_v));

#pragma omp parallel for simd
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];

    E_arr[z][y][x][X] = arr_v[z][y][x][0];
    E_arr[z][y][x][Y] = arr_v[z][y][x][1];
    E_arr[z][y][x][Z] = arr_v[z][y][x][2];

    B_arr[z][y][x][X] = arr_v[z][y][x][3];
    B_arr[z][y][x][Y] = arr_v[z][y][x][4];
    B_arr[z][y][x][Z] = arr_v[z][y][x][5];
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da_EB, v, &arr_v));
  PetscCall(DMDAVecRestoreArrayWrite(da, E_hk, &E_arr));
  PetscCall(DMDAVecRestoreArrayWrite(da, B_hk, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Simulation::to_snes(Vec vE, Vec vB, Vec v)
{
  PetscFunctionBeginUser;
  PetscReal**** arr_v;
  PetscCall(DMDAVecGetArrayRead(da, vE, &E_arr));
  PetscCall(DMDAVecGetArrayRead(da, vB, &B_arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da_EB, v, &arr_v));

#pragma omp parallel for simd
  for (PetscInt g = 0; g < world.size.elements_product(); ++g) {
    PetscInt x = world.start[X] + g % world.size[X];
    PetscInt y = world.start[Y] + (g / world.size[X]) % world.size[Y];
    PetscInt z = world.start[Z] + (g / world.size[X]) / world.size[Y];
    arr_v[z][y][x][0] = E_arr[z][y][x][X];
    arr_v[z][y][x][1] = E_arr[z][y][x][Y];
    arr_v[z][y][x][2] = E_arr[z][y][x][Z];

    arr_v[z][y][x][3] = B_arr[z][y][x][X];
    arr_v[z][y][x][4] = B_arr[z][y][x][Y];
    arr_v[z][y][x][5] = B_arr[z][y][x][Z];
  }

  PetscCall(DMDAVecRestoreArrayDOFWrite(da_EB, v, &arr_v));
  PetscCall(DMDAVecRestoreArrayRead(da, E_hk, &E_arr));
  PetscCall(DMDAVecRestoreArrayRead(da, B_hk, &B_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}


EnergyConservation::EnergyConservation(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/dk_diagnostic.txt"),
    simulation(simulation)
{
}

PetscErrorCode EnergyConservation::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (charge_da == nullptr && t != 0) {
    PetscCall(init_charge_conservation());
    PetscCall(collect_charge_densities());
  }

  if (!initialized) {
    PetscCall(VecNorm(simulation.E, NORM_2, &w_E));
    PetscCall(VecNorm(simulation.B, NORM_2, &w_B));
    PetscCall(VecDot(simulation.M, simulation.B, &a_MB));
    w_E = 0.5 * POW2(w_E);
    w_B = 0.5 * POW2(w_B);
    K = simulation.particles_[0]->kinetic_energy_local();
    initialized = true;
  }

  w_E0 = w_E;
  w_B0 = w_B;
  a_MB0 = a_MB;
  K0 = K;
  PetscCall(TableDiagnostic::diagnose(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::initialize()
{
  PetscFunctionBeginUser;
  PetscCall(init_charge_conservation());
  PetscCall(collect_charge_densities());
  PetscCall(TableDiagnostic::initialize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::finalize()
{
  PetscFunctionBeginUser;
  for (auto& vec : charge_locals)
    PetscCall(VecDestroy(&vec));
  for (auto& vec : charge_fields)
    PetscCall(VecDestroy(&vec));

  charge_locals.clear();
  charge_fields.clear();
  current_densities.clear();

  PetscCall(MatDestroy(&divE));
  PetscCall(DMDestroy(&charge_da));
  PetscCall(TableDiagnostic::finalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::init_charge_conservation()
{
  PetscFunctionBeginUser;
  if (charge_da != nullptr)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt g_size[3];
  PetscInt procs[3];
  PetscInt s;
  DMBoundaryType bounds[3];
  DMDAStencilType st;
  PetscCall(DMDAGetInfo(simulation.da, nullptr, REP3_A(&g_size), REP3_A(&procs), nullptr,
    &s, REP3_A(&bounds), &st));

  const PetscInt* ownership[3];
  PetscCall(DMDAGetOwnershipRanges(simulation.da, REP3_A(&ownership)));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, REP3_A(bounds), st, REP3_A(g_size),
    REP3_A(procs), 1, s, REP3_A(ownership), &charge_da));
  PetscCall(DMSetUp(charge_da));

  current_densities.reserve(simulation.particles_.size() + 1);
  charge_locals.reserve(simulation.particles_.size());
  charge_fields.reserve(simulation.particles_.size());

  for (const auto& sort : simulation.particles_) {
    Vec local = nullptr;
    Vec field = nullptr;

    PetscCall(DMCreateLocalVector(charge_da, &local));
    PetscCall(DMCreateGlobalVector(charge_da, &field));

    charge_locals.emplace_back(local);
    charge_fields.emplace_back(field);
    current_densities.emplace_back(sort->J);
  }
  current_densities.emplace_back(simulation.J);

  PetscCheckAbort(current_densities.size() == charge_fields.size() + 1, PETSC_COMM_WORLD,
    PETSC_ERR_USER,
    "Number of `current_densities` should be one more than charge fields");

  Divergence divergence(simulation.da);
  PetscCall(divergence.create_negative(&divE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::collect_charge_density(PetscInt sort_id)
{
  PetscFunctionBeginUser;
  PetscCheck(sort_id >= 0 &&
      sort_id < (PetscInt)simulation.particles_.size(), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Invalid drift-kinetic sort index %" PetscInt_FMT, sort_id);

  auto& local = charge_locals[sort_id];
  auto& field = charge_fields[sort_id];
  const auto& sort = *simulation.particles_[sort_id];
  const interfaces::Particles& particles = sort;
  const auto& dk_curr_storage = sort.get_dk_curr_storage();

  PetscCall(VecSet(local, 0.0));
  PetscCall(VecSet(field, 0.0));

  PetscReal*** arr = nullptr;
  PetscCall(DMDAVecGetArrayWrite(charge_da, local, &arr));

  ChargeShape shape;

#pragma omp parallel for private(shape)
  for (PetscInt g = 0; g < (PetscInt)dk_curr_storage.size(); ++g) {
    for (const auto& point : dk_curr_storage[g]) {
      shape.setup(point.r);

      Point equivalent_point(point.r, Vector3R{});
      const PetscReal qn_np = particles.qn_Np(equivalent_point);

      for (PetscInt i = 0; i < shape.shm; ++i) {
        const PetscInt g_x = shape.start[X] + i % shape.shw;
        const PetscInt g_y = shape.start[Y] + (i / shape.shw) % shape.shw;
        const PetscInt g_z = shape.start[Z] + (i / shape.shw) / shape.shw;

#pragma omp atomic update
        arr[g_z][g_y][g_x] += qn_np * shape.cache[i];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayWrite(charge_da, local, &arr));
  PetscCall(DMLocalToGlobal(charge_da, local, ADD_VALUES, field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::collect_charge_densities()
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < (PetscInt)simulation.particles_.size(); ++i)
    PetscCall(collect_charge_density(i));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode EnergyConservation::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);
  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    add(16, "AvgPushIt_" + name, "{: .3f}", sort->get_average_iteration_number());
    add(16, "AvgCells_" + name, "{: .3f}", sort->get_average_number_of_traversed_cells());
  }
  add(10, "AvgFieldIt", "{:d}", simulation.last_field_itnum);

  PetscCall(VecNorm(simulation.E, NORM_2, &w_E));
  PetscCall(VecNorm(simulation.B, NORM_2, &w_B));
  PetscCall(VecNorm(simulation.M, NORM_2, &w_M));
  PetscCall(VecNorm(simulation.Mn, NORM_2, &w_Mn));
  PetscCall(VecDot(simulation.E_hk, simulation.J, &a_EJ));
  PetscCall(VecDot(simulation.M, simulation.B, &a_MB));
  w_E = 0.5 * POW2(w_E);
  w_B = 0.5 * POW2(w_B);
  K = simulation.particles_[0]->kinetic_energy_local();

  dF = (w_E - w_E0) + (w_B - w_B0);

  add(13, "dK", "{: .6e}", (K - K0));
  add(13, "dE", "{: .6e}", (w_E - w_E0));
  add(13, "dB", "{: .6e}", (w_B - w_B0));
  add(13, "dE+dB+dK", "{: .6e}", dF + (K - K0));
  //add(13, "dMB", "{: .6e}", (a_MB - a_MB0));
  //add(13, "dt * a_EJ", "{: .6e}", (dt * a_EJ));
  add(16, "dE+dB-dMB+dt*dEJ", "{: .6e}", dF - (a_MB - a_MB0) + dt * a_EJ);
  //add(16, "dM/dt-mu*VgradB", "{: .6e}", (w_M - w_Mn)/dt - simulation.VgradB);

  Vec sum = nullptr;
  Vec diff = nullptr;
  PetscReal norm[2];

  PetscCall(DMGetGlobalVector(charge_da, &diff));
  PetscCall(DMGetGlobalVector(charge_da, &sum));
  PetscCall(VecSet(sum, 0.0));

  PetscInt i = 0;
  for (; i < (PetscInt)charge_fields.size(); ++i) {
    PetscCall(VecCopy(charge_fields[i], diff));
    PetscCall(collect_charge_density(i));
    PetscCall(VecAYPX(diff, -1.0, charge_fields[i]));
    PetscCall(VecScale(diff, 1.0 / dt));

    PetscCall(VecAXPY(sum, 1.0, diff));

    PetscCall(MatMultAdd(divE, current_densities[i], diff, diff));
    PetscCall(VecNorm(diff, NORM_1_AND_2, norm));

    const auto& name = simulation.particles_[i]->parameters.sort_name;
    add(13, "N1dQ_" + name, "{: .6e}", norm[0]);
    add(13, "N2dQ_" + name, "{: .6e}", norm[1]);
  }

  PetscCall(MatMultAdd(divE, current_densities[i], sum, sum));
  PetscCall(VecNorm(sum, NORM_1_AND_2, norm));

  add(13, "N1dQ_tot", "{: .6e}", norm[0]);
  add(13, "N2dQ_tot", "{: .6e}", norm[1]);

  PetscCall(DMRestoreGlobalVector(charge_da, &sum));
  PetscCall(DMRestoreGlobalVector(charge_da, &diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace drift_kinetic
