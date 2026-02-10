#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test: magnetic mirror with double-Gaussian mirrors.\n"
  "Guiding-center push (analytic/grid) should match the Boris pusher.\n";

using namespace drift_kinetic_test_utils;
using namespace gaussian_magnetic_mirror;

namespace {

constexpr PetscReal segment_eps = 1e-14;

constexpr PetscReal tol_E_interp = 1e-8;
constexpr PetscReal tol_B_interp = 5e-3;
constexpr PetscReal tol_gradB_interp = 5e-2;

constexpr PetscReal tol_z = 5e-2;
constexpr PetscReal tol_par = 5e-2;
constexpr PetscReal tol_mu = 5e-2;
constexpr PetscReal tol_energy = 5e-2;

void eval_point_field(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  get_fields(r, r, E_p, B_p, gradB_p);
  E_p = {0.0, 0.0, 0.0};
}

void eval_grid_field_yee(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  Vector3R E_ex, E_ey, E_ez;
  Vector3R B_ex, B_ey, B_ez;
  Vector3R E_bx, E_by, E_bz;
  Vector3R B_bx, B_by, B_bz;
  Vector3R grad_dummy;

  eval_point_field(implicit_test_utils::yee_pos_ex(r), E_ex, B_ex, grad_dummy);
  eval_point_field(implicit_test_utils::yee_pos_ey(r), E_ey, B_ey, grad_dummy);
  eval_point_field(implicit_test_utils::yee_pos_ez(r), E_ez, B_ez, grad_dummy);

  eval_point_field(implicit_test_utils::yee_pos_bx(r), E_bx, B_bx, grad_dummy);
  eval_point_field(implicit_test_utils::yee_pos_by(r), E_by, B_by, grad_dummy);
  eval_point_field(implicit_test_utils::yee_pos_bz(r), E_bz, B_bz, grad_dummy);

  E_p = {E_ex.x(), E_ey.y(), E_ez.z()};
  B_p = {B_bx.x(), B_by.y(), B_bz.z()};

  Vector3R E_center, B_center;
  eval_point_field(r, E_center, B_center, gradB_p);
}

}  // namespace

void get_analytical_fields(const Vector3R&, const Vector3R& rn, Vector3R& E_p,
  Vector3R& B_p, Vector3R& gradB_p)
{
  eval_point_field(rn, E_p, B_p, gradB_p);
}

void get_grid_fields(const Vector3R&, const Vector3R& rn, Vector3R& E_p,
  Vector3R& B_p, Vector3R& gradB_p)
{
  eval_grid_field_yee(rn, E_p, B_p, gradB_p);
} 

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscReal omega_dt;
  PetscCall(get_omega_dt(omega_dt));

  dx = 0.01;
  dt = omega_dt / B_min;
  geom_nx = (PetscInt)(2 * L / dx);

  overwrite_config(2 * Rc, 2 * Rc, 2 * L, 3000*dt, dx, dx, dx, dt, dt);

  FieldContext context;

  PetscCall(context.initialize([&](PetscInt i, PetscInt j, PetscInt k, Vector3R& E_g, Vector3R& B_g, Vector3R& gradB_g) {
    Vector3R r(i * dx, j * dy, k * dz);
    get_grid_fields(r, r, E_g, B_g, gradB_g);
  }));

  DriftKineticEsirkepov esirkepov(
    context.E_arr, context.B_arr, nullptr, nullptr);

  esirkepov.set_dBidrj(context.dBdx_arr, context.dBdy_arr, context.dBdz_arr);

  constexpr PetscReal v_par = 0.01 / M_SQRT2;
  constexpr PetscReal v_perp = 0.01 / M_SQRT2;
  constexpr Vector3R r0(Rc + 0.02, Rc, L);
  constexpr Vector3R v0(v_perp, 0.0, v_par);
  Point point_init(r0, v0);

  auto f_anal = [&](const Vector3R& r0, const Vector3R& rn, Vector3R& E_p,
      Vector3R& B_p, Vector3R& gradB_p){
        E_p = {};
        B_p = {};
        gradB_p = {};
        Vector3R Es_p, Bs_p, gradBs_p;

        Vector3R pos = (rn - r0);
        PetscReal path = pos.length();

        auto coords = cell_traversal(rn, r0);
        PetscInt segments = (PetscInt)coords.size() - 1;
        if (segments <= 0) {
          segments = 1;
        }

        pos[X] = pos[X] != 0 ? pos[X] : (PetscReal)segments;
        pos[Y] = pos[Y] != 0 ? pos[Y] : (PetscReal)segments;
        pos[Z] = pos[Z] != 0 ? pos[Z] : (PetscReal)segments;

        Vector3R E_dummy, gradB_dummy;
        eval_point_field(rn, E_dummy, B_p, gradB_dummy);

      for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
        auto&& rs0 = coords[s - 1];
        auto&& rsn = coords[s - 0];

        Vector3R dseg = rsn - rs0;
        PetscReal seg_path = dseg.length();
        if (seg_path < segment_eps) {
          continue;
        }

        get_analytical_fields(rs0, rsn, Es_p, Bs_p, gradBs_p);

        Vector3R drs{
          rsn[X] != rs0[X] ? rsn[X] - rs0[X] : 1.0,
          rsn[Y] != rs0[Y] ? rsn[Y] - rs0[Y] : 1.0,
          rsn[Z] != rs0[Z] ? rsn[Z] - rs0[Z] : 1.0,
        };

        PetscReal beta = path > 0 ? seg_path / path : 1.0;
        E_p += Es_p * beta;
        gradB_p += gradBs_p.elementwise_product(drs).elementwise_division(pos);
      }
      };

    auto f_grid = [&](const Vector3R& r0, const Vector3R& rn, //
      Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p) {
      E_p = {};
      B_p = {};
      gradB_p = {};
      Vector3R Es_p, Bs_p, gradBs_p;

      Vector3R pos = (rn - r0);
      PetscReal path = pos.length();

      auto coords = cell_traversal(rn, r0);
      PetscInt segments = (PetscInt)coords.size() - 1;
      if (segments <= 0) {
        segments = 1;
      }

      pos[X] = pos[X] != 0 ? pos[X] / dx : (PetscReal)segments;
      pos[Y] = pos[Y] != 0 ? pos[Y] / dy : (PetscReal)segments;
      pos[Z] = pos[Z] != 0 ? pos[Z] / dz : (PetscReal)segments;

      Vector3R E_dummy, gradB_dummy;
      esirkepov.interpolate(E_dummy, B_p, gradB_dummy, rn, r0);

      for (PetscInt s = 1; s < (PetscInt)coords.size(); ++s) {
        auto&& rs0 = coords[s - 1];
        auto&& rsn = coords[s - 0];

        Vector3R dseg = rsn - rs0;
        PetscReal seg_path = dseg.length();
        if (seg_path < segment_eps) {
          continue;
        }

        esirkepov.interpolate(Es_p, Bs_p, gradBs_p, rsn, rs0);

        PetscReal beta = path > 0 ? seg_path / path : 1.0;
        E_p += Es_p * beta;
        gradB_p += gradBs_p.elementwise_division(pos);
      }
    };

  Vector3R E0, B0, gradB0;
  eval_point_field(r0, E0, B0, gradB0);

  Vector3R B0_anal = B0;
  PointByField point_analytical(point_init, B0_anal, m, q / m);
  Vector3R B0_grid = B0;
  PointByField point_grid(point_init, B0_grid, m, q / m);
  Point point_boris(point_init);

  ComparisonStats stats;
  stats.ref_mu = point_grid.mu();
  stats.ref_energy = get_kinetic_energy(point_grid);

  LOG("B0_anal: ({:.6e} {:.6e} {:.6e}) |B0_anal|={:.6e}", REP3_A(B0_anal), B0_anal.length());
  LOG("B0_grid: ({:.6e} {:.6e} {:.6e}) |B0_grid|={:.6e}", REP3_A(B0_grid), B0_grid.length());
  LOG("mu_anal={:.6e} mu_grid={:.6e}", point_analytical.mu(), point_grid.mu());

  DriftKineticPush push_analytical;
  push_analytical.set_qm(q / m);
  push_analytical.set_mp(m);
  push_analytical.set_fields_callback(f_anal);

  DriftKineticPush push_grid;
  push_grid.set_qm(q / m);
  push_grid.set_mp(m);
  push_grid.set_fields_callback(f_grid);

  BorisPush push_boris;
  push_boris.set_qm(q / m);

  TraceTriplet trace(__FILE__, std::format("omega_dt_{:.4f}", omega_dt),
    1., point_analytical, point_grid, point_boris);

  Vector3R E_analytical, B_analytical, gradB_analytical;
  Vector3R E_grid, B_grid, gradB_grid;

  for (PetscInt t = 0; t <= geom_nt; ++t) {
    const PointByField point_analytical_old = point_analytical;
    const PointByField point_grid_old = point_grid;

    PetscCall(trace.diagnose(t));

    push_analytical.process(dt, point_analytical, point_analytical_old);
    push_grid.process(dt, point_grid, point_grid_old);
    boris_step(push_boris, point_boris, get_analytical_fields);

    PetscCheck(std::abs(point_analytical.z() - L) <= L, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_analytical.z() - L, L);

    PetscCheck(std::abs(point_grid.z() - L) <= L, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Grid particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_grid.z() - L, L);

    PetscCheck(std::abs(point_boris.z() - L) <= L, PETSC_COMM_WORLD, PETSC_ERR_USER,
      "Boris particle must not escape magnetic mirror! z = %.6e, allowed max = %.6e", point_boris.z() - L, L);

    f_anal(point_analytical_old.r, point_analytical.r,
      E_analytical, B_analytical, gradB_analytical);

    f_grid(point_grid_old.r, point_grid.r, E_grid, B_grid, gradB_grid);

    {
      const Vector3R probe_r0 = point_grid_old.r;
      const Vector3R probe_rn = point_grid.r;
      Vector3R E_probe_anal, B_probe_anal, gradB_probe_anal;
      Vector3R E_probe_grid, B_probe_grid, gradB_probe_grid;

      f_anal(probe_r0, probe_rn, E_probe_anal, B_probe_anal, gradB_probe_anal);
      f_grid(probe_r0, probe_rn, E_probe_grid, B_probe_grid, gradB_probe_grid);

      PetscCheck(equal_tol(E_probe_grid, E_probe_anal, tol_E_interp), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Segment E mismatch for identical trajectory. Analytical: (%.8e %.8e %.8e), grid: (%.8e %.8e %.8e)",
        REP3_A(E_probe_anal), REP3_A(E_probe_grid));

      PetscCheck(equal_tol(B_probe_grid, B_probe_anal, tol_B_interp), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Segment B mismatch for identical trajectory. Analytical: (%.8e %.8e %.8e), grid: (%.8e %.8e %.8e)",
        REP3_A(B_probe_anal), REP3_A(B_probe_grid));

      PetscCheck(equal_tol(gradB_probe_grid, gradB_probe_anal, tol_gradB_interp), PETSC_COMM_WORLD, PETSC_ERR_USER,
        "Segment gradB mismatch for identical trajectory. Analytical: (%.8e %.8e %.8e), grid: (%.8e %.8e %.8e)",
        REP3_A(gradB_probe_anal), REP3_A(gradB_probe_grid));
    }

    if (t < 20) {
      Vector3R b_analytical = B_analytical.normalized();
      Vector3R b_grid = B_grid.normalized();

      PetscReal bgradB_analytical = b_analytical.dot(gradB_analytical);
      PetscReal bgradB_grid = b_grid.dot(gradB_grid);

      PetscReal Vh_analytical = 0.5 * (point_analytical_old.p_parallel + point_analytical.p_parallel);
      PetscReal Vh_grid = 0.5 * (point_grid_old.p_parallel + point_grid.p_parallel);

      LOG("t={} b·gradB: anal={:.6e} grid={:.6e} b·Vp(=Vh): anal={:.6e} grid={:.6e}",
        t, bgradB_analytical, bgradB_grid, Vh_analytical, Vh_grid);
    }

    update_comparison_stats(stats,  //
      point_analytical, point_grid, point_boris,  //
      B_analytical, gradB_analytical, B_grid, gradB_grid);
  }

  PetscCheck(stats.max_err_z <= tol_z, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift vs Boris z mismatch too high: max_err_z=%.8e, tol=%.8e", stats.max_err_z, tol_z);

  PetscCheck(stats.max_err_par <= tol_par, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift vs Boris p_parallel mismatch too high: max_err_par=%.8e, tol=%.8e", stats.max_err_par, tol_par);

  PetscCheck(stats.max_err_mu <= tol_mu, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift vs Boris mu mismatch too high: max_err_mu=%.8e, tol=%.8e", stats.max_err_mu, tol_mu);

  PetscCheck(stats.max_err_energy <= tol_energy, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "Drift vs Boris energy mismatch too high: max_err_energy=%.8e, tol=%.8e",
    stats.max_err_energy, tol_energy);

  PetscCall(print_statistics(stats, point_analytical, point_grid, point_boris));

  PetscCall(context.finalize());
  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
