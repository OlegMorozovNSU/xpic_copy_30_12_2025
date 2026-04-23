#include "src/impls/drift_kinetic/simulation.h"
#include "src/utils/configuration.h"
#include "tests/common.h"

static constexpr char help[] =
  "Test of energy and charge conservation for \"drift_kinetic\" implementation.  \n"
  "The simplest case is tested: plasma cube of size L=5.0 (N=10) is modeled \n"
  "in periodic boundaries for 100 cycles (dt=1.5). There are only maxwellian\n"
  "electrons with the temperature T=100 eV, ions are stationary background. \n";

void overwrite_config();

int main(int argc, char** argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, nullptr, help));

  overwrite_config();
  Configuration::save(get_out_dir(__FILE__));

  drift_kinetic::Simulation simulation;
  PetscCall(simulation.initialize());
  PetscCall(simulation.calculate());
  PetscCall(simulation.finalize());

  //PetscCall(compare_temporal(__FILE__, "energy_conservation.txt"));
  //PetscCall(compare_temporal(__FILE__, "charge_conservation.txt"));

  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}

void overwrite_config()
{
  dx = 10;
  geom_ny = 5;
  geom_y = geom_ny * dx;
  geom_nx = 100;
  geom_x = geom_nx * dx;
  geom_nz = 100;
  geom_z = geom_nz * dx;

  dt = 1;
  geom_nt = 10000;
  geom_t = geom_nt * dt;

  Configuration::overwrite({
    {"Simulation", "drift_kinetic"},
    {"OutputDirectory", get_out_dir(__FILE__)},
    {
      "Geometry",
      {
        {"x", geom_x},
        {"y", geom_y},
        {"z", geom_z},
        {"t", geom_t},
        {"dx", dx},
        {"dy", dx},
        {"dz", dx},
        {"dt", dt},
        {"diagnose_period", 2*dt},
        {"da_boundary_x", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_y", "DM_BOUNDARY_PERIODIC"},
        {"da_boundary_z", "DM_BOUNDARY_PERIODIC"},
      },
    },
    {
      "Particles",
      {{
        {"sort_name", "electrons"},
        {"Np", 1},
        {"n", +1.0},
        {"q", -1.0},
        {"m", +1.0},
        {"T", +0.1},
      },
      {
        {"sort_name", "ions"},
        {"Np", 1},
        {"n", +1.0},
        {"q", +1.0},
        {"m", +1836.0},
        {"T", +0.1},
      }},
    },
    {
      "Presets",
      {
        {
          {"command", "SetMagneticField"},
          {"field", "B0"},
          {"field_axpy", "B"},
          {
            "setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, 0.394}},
            },
          },
        },
        {
          {"command", "SetMagneticField"},
          {"field", "B"},
          {
            "setter",
            {
              {"name", "SetGeneralCosineField"},
              {"amplitude", {-0.079, 0.0, 0.0}},
              {"wave_number", {0.0, 0.0, 1.0}},
              {"min", {0.0, 0.0, 0.0}},
              {"max", {"geom_x", "geom_y", "geom_z"}},
            },
          },
        },
        {
          {"command", "SetElectricField"},
          {
            "setter",
            {
              {"name", "SetUniformField"},
              {"value", {0.0, 0.0, 0.0}},
            },
          },
        },
        {
          {"command", "SetElectricField"},
          {"field", "E"},
          {
            "setter",
            {
              {"name", "SetGeneralCosineField"},
              {"amplitude", {0.0, 0.394 * 1.84e-3, 0.0}},
              {"wave_number", {0.0, 0.0, 1.0}},
            },
          },
        },
        {
          {"command", "SetParticles"},
          {"particles", "electrons"},
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum", {{"name", "MaxwellianMomentum"}, {"tov", true}}},
        },
        {
          {"command", "SetParticles"},
          {"particles", "ions"},
          {"coordinate", {{"name", "CoordinateInBox"}}},
          {"momentum",
            {
              {"name", "GeneralCosineVelocity"},
              {"amplitude", {1.84e-3, 0.0, 0.0}},
              {"wave_number", {0.0, 0.0, 1.0}}
            }
          },
        },
      },
    },
    {
      "Diagnostics",
  {
    {
      {"diagnostic", "FieldView"},
      {"field", "E"},
      {"region", {{"type", "2D"}, {"plane", "X"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "E"},
      {"region", {{"type", "2D"}, {"plane", "Y"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "E"},
      {"region", {{"type", "2D"}, {"plane", "Z"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "B"},
      {"region", {{"type", "2D"}, {"plane", "X"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "B"},
      {"region", {{"type", "2D"}, {"plane", "Y"}}},
    },
    {
      {"diagnostic", "FieldView"},
      {"field", "B"},
      {"region", {{"type", "2D"}, {"plane", "Z"}}},
    },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "electrons"},
          {"moment", "density"},
        },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "ions"},
          {"moment", "density"},
        },
                {
          {"diagnostic", "DistributionMoment"},
          {"particles", "electrons"},
          {"moment", "current"},
        },
        {
          {"diagnostic", "DistributionMoment"},
          {"particles", "ions"},
          {"moment", "current"},
        },
      },
    },
  });
}
