#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov interpolation: ex2 cyclic profile family.\n"
  "Covers baseline, linear mixed, and nonlinear polynomial fields.\n";

using namespace implicit_test_utils;

namespace {

constexpr Vector3R E0(1.0, 1.0, 1.0);
constexpr PetscReal eps_nl = 1e-6;

const InterpTolerance nonlinear_tol{
  .E = 1e-5,
  .B = 1e-5,
  .gradB = 1e-5,
};

Vector3R eval_baseline_B(const Vector3R& r)
{
  return {r.y(), r.z(), r.x()};
}

Vector3R eval_linear_E(const Vector3R& r)
{
  return {
    0.2 + 0.3 * r.y() + 0.1 * r.z() - 0.05 * r.x(),
    -0.1 + 0.04 * r.y() + 0.2 * r.z() + 0.05 * r.x(),
    0.4 + 0.07 * r.y() - 0.03 * r.z() - 0.15 * r.x(),
  };
}

Vector3R eval_linear_B(const Vector3R& r)
{
  return {
    1.0 + 0.2 * r.y() + 0.4 * r.z(),
    0.7 - 0.3 * r.z() + 0.25 * r.x(),
    1.3 - 0.2 * r.y() + 0.1 * r.x(),
  };
}

Vector3R eval_linear_gradB(const Vector3R& r)
{
  Vector3R B = eval_linear_B(r);
  Vector3R dBdx{0.0, 0.25, 0.1};
  Vector3R dBdy{0.2, 0.0, -0.2};
  Vector3R dBdz{0.4, -0.3, 0.0};
  return grad_abs_b(B, dBdx, dBdy, dBdz);
}

Vector3R eval_nonlinear_E(const Vector3R& r)
{
  return {
    0.1 + 0.05 * r.y() + eps_nl * (r.y() * r.z() + r.x() * r.x()),
    -0.2 + 0.04 * r.z() + eps_nl * (r.z() * r.x() + r.y() * r.y()),
    0.3 - 0.03 * r.x() + eps_nl * (r.x() * r.y() + r.z() * r.z()),
  };
}

Vector3R eval_nonlinear_B(const Vector3R& r)
{
  return {
    1.2 + 0.15 * r.y() + eps_nl * (r.y() * r.y() + 0.5 * r.z() * r.x()),
    0.9 - 0.12 * r.z() + eps_nl * (r.z() * r.z() + 0.5 * r.y() * r.x()),
    1.5 + 0.08 * r.x() + eps_nl * (r.x() * r.x() + 0.5 * r.y() * r.z()),
  };
}

Vector3R eval_nonlinear_gradB(const Vector3R& r)
{
  Vector3R B = eval_nonlinear_B(r);
  Vector3R dBdx{
    0.5 * eps_nl * r.z(),
    0.5 * eps_nl * r.y(),
    0.08 + 2.0 * eps_nl * r.x(),
  };
  Vector3R dBdy{
    0.15 + 2.0 * eps_nl * r.y(),
    0.5 * eps_nl * r.x(),
    0.5 * eps_nl * r.z(),
  };
  Vector3R dBdz{
    0.5 * eps_nl * r.x(),
    -0.12 + 2.0 * eps_nl * r.z(),
    0.5 * eps_nl * r.y(),
  };
  return grad_abs_b(B, dBdx, dBdy, dBdz);
}

void baseline_analytic(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = eval_baseline_B(r);
  gradB_p = grad_abs_b(B_p, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
}

void baseline_grid(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = E0;
  B_p = {
    eval_baseline_B(yee_pos_bx(r)).x(),
    eval_baseline_B(yee_pos_by(r)).y(),
    eval_baseline_B(yee_pos_bz(r)).z(),
  };
  gradB_p = grad_abs_b(eval_baseline_B(r), {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
}

void linear_analytic(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = eval_linear_E(r);
  B_p = eval_linear_B(r);
  gradB_p = eval_linear_gradB(r);
}

void linear_grid(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = {
    eval_linear_E(yee_pos_ex(r)).x(),
    eval_linear_E(yee_pos_ey(r)).y(),
    eval_linear_E(yee_pos_ez(r)).z(),
  };
  B_p = {
    eval_linear_B(yee_pos_bx(r)).x(),
    eval_linear_B(yee_pos_by(r)).y(),
    eval_linear_B(yee_pos_bz(r)).z(),
  };
  gradB_p = eval_linear_gradB(r);
}

void nonlinear_analytic(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = eval_nonlinear_E(r);
  B_p = eval_nonlinear_B(r);
  gradB_p = eval_nonlinear_gradB(r);
}

void nonlinear_grid(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = {
    eval_nonlinear_E(yee_pos_ex(r)).x(),
    eval_nonlinear_E(yee_pos_ey(r)).y(),
    eval_nonlinear_E(yee_pos_ez(r)).z(),
  };
  B_p = {
    eval_nonlinear_B(yee_pos_bx(r)).x(),
    eval_nonlinear_B(yee_pos_by(r)).y(),
    eval_nonlinear_B(yee_pos_bz(r)).z(),
  };
  gradB_p = eval_nonlinear_gradB(r);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscCall(run_interpolation_profile(
    "ex2_baseline", interpolation_full_trajectories(), baseline_analytic, baseline_grid));

  PetscCall(run_interpolation_profile(
    "ex2_linear_mixed", interpolation_full_trajectories(), linear_analytic, linear_grid));

  PetscCall(run_interpolation_profile(
    "ex2_nonlinear_poly", interpolation_nonlinear_trajectories(),
    nonlinear_analytic, nonlinear_grid, nonlinear_tol));

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
