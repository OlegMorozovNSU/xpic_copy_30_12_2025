#include "drift_kinetic_push.h"

static constexpr char help[] =
  "Test drift-kinetic Esirkepov interpolation: ex4 dense regression.\n"
  "Covers affine mixed and nonlinear polynomial mixed field profiles.\n";

using namespace implicit_test_utils;

namespace {

constexpr PetscReal eps_poly = 1e-6;

const InterpTolerance nonlinear_tol{
  .E = 1e-5,
  .B = 1e-5,
  .gradB = 1e-5,
};

Vector3R eval_affine_E(const Vector3R& r)
{
  return {
    0.2 + 0.11 * r.x() - 0.07 * r.y() + 0.05 * r.z(),
    -0.1 + 0.08 * r.x() + 0.09 * r.y() - 0.04 * r.z(),
    0.3 - 0.06 * r.x() + 0.02 * r.y() + 0.07 * r.z(),
  };
}

Vector3R eval_affine_B(const Vector3R& r)
{
  return {
    1.0 + 0.15 * r.x() - 0.12 * r.y() + 0.09 * r.z(),
    0.8 - 0.05 * r.x() + 0.13 * r.y() + 0.11 * r.z(),
    1.4 + 0.07 * r.x() - 0.03 * r.y() + 0.16 * r.z(),
  };
}

Vector3R eval_affine_gradB(const Vector3R& r)
{
  Vector3R B = eval_affine_B(r);
  Vector3R dBdx{0.15, -0.05, 0.07};
  Vector3R dBdy{-0.12, 0.13, -0.03};
  Vector3R dBdz{0.09, 0.11, 0.16};
  return grad_abs_b(B, dBdx, dBdy, dBdz);
}

Vector3R eval_poly_E(const Vector3R& r)
{
  return {
    0.25 + 0.1 * r.x() - 0.05 * r.y() + 0.07 * r.z() +
      eps_poly * (r.x() * r.y() + r.y() * r.z() + r.z() * r.x() + r.x() * r.x()),
    -0.15 + 0.09 * r.x() + 0.04 * r.y() - 0.06 * r.z() +
      eps_poly * (r.x() * r.z() + r.y() * r.y() + r.z() * r.z() + r.x() * r.y()),
    0.35 - 0.08 * r.x() + 0.03 * r.y() + 0.05 * r.z() +
      eps_poly * (r.x() * r.x() + r.y() * r.z() + r.z() * r.x() + r.y() * r.y()),
  };
}

Vector3R eval_poly_B(const Vector3R& r)
{
  return {
    1.1 + 0.12 * r.x() - 0.09 * r.y() + 0.06 * r.z() +
      eps_poly * (r.x() * r.y() + r.y() * r.z() + r.z() * r.z()),
    0.95 - 0.07 * r.x() + 0.1 * r.y() + 0.08 * r.z() +
      eps_poly * (r.x() * r.z() + r.x() * r.x() + r.y() * r.z()),
    1.55 + 0.05 * r.x() - 0.04 * r.y() + 0.14 * r.z() +
      eps_poly * (r.x() * r.y() + r.y() * r.y() + r.z() * r.x()),
  };
}

Vector3R eval_poly_gradB(const Vector3R& r)
{
  Vector3R B = eval_poly_B(r);
  Vector3R dBdx{
    0.12 + eps_poly * r.y(),
    -0.07 + eps_poly * (r.z() + 2.0 * r.x()),
    0.05 + eps_poly * (r.y() + r.z()),
  };
  Vector3R dBdy{
    -0.09 + eps_poly * (r.x() + r.z()),
    0.1 + eps_poly * r.z(),
    -0.04 + eps_poly * (r.x() + 2.0 * r.y()),
  };
  Vector3R dBdz{
    0.06 + eps_poly * (r.y() + 2.0 * r.z()),
    0.08 + eps_poly * (r.x() + r.y()),
    0.14 + eps_poly * r.x(),
  };
  return grad_abs_b(B, dBdx, dBdy, dBdz);
}

void affine_analytic(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = eval_affine_E(r);
  B_p = eval_affine_B(r);
  gradB_p = eval_affine_gradB(r);
}

void affine_grid(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = {
    eval_affine_E(yee_pos_ex(r)).x(),
    eval_affine_E(yee_pos_ey(r)).y(),
    eval_affine_E(yee_pos_ez(r)).z(),
  };
  B_p = {
    eval_affine_B(yee_pos_bx(r)).x(),
    eval_affine_B(yee_pos_by(r)).y(),
    eval_affine_B(yee_pos_bz(r)).z(),
  };
  gradB_p = eval_affine_gradB(r);
}

void poly_analytic(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = eval_poly_E(r);
  B_p = eval_poly_B(r);
  gradB_p = eval_poly_gradB(r);
}

void poly_grid(
  const Vector3R& r, Vector3R& E_p, Vector3R& B_p, Vector3R& gradB_p)
{
  E_p = {
    eval_poly_E(yee_pos_ex(r)).x(),
    eval_poly_E(yee_pos_ey(r)).y(),
    eval_poly_E(yee_pos_ez(r)).z(),
  };
  B_p = {
    eval_poly_B(yee_pos_bx(r)).x(),
    eval_poly_B(yee_pos_by(r)).y(),
    eval_poly_B(yee_pos_bz(r)).z(),
  };
  gradB_p = eval_poly_gradB(r);
}

}  // namespace

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  PetscCall(run_interpolation_profile(
    "ex4_affine_mixed", interpolation_full_trajectories(), affine_analytic, affine_grid));

  PetscCall(run_interpolation_profile(
    "ex4_nonlinear_mixed", interpolation_nonlinear_trajectories(),
    poly_analytic, poly_grid, nonlinear_tol));

  PetscCall(PetscFinalize());
  return EXIT_SUCCESS;
}
