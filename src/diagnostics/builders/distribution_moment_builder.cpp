#include "distribution_moment_builder.h"

#include "src/diagnostics/distribution_moment.h"
#include "src/utils/utils.h"
#include "src/utils/vector_utils.h"

DistributionMomentBuilder::DistributionMomentBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : FieldViewBuilder(simulation, diagnostics)
{
}

PetscErrorCode DistributionMomentBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  const std::map<std::string_view, PetscInt> available_moments{
    {"density", 1},
    {"current", 3},
    {"momentum_flux", 6},
    {"momentum_flux_cyl", 6},
    {"momentum_flux_diag", 3},
    {"momentum_flux_diag_cyl", 3},
  };

  std::string particles;
  info.at("particles").get_to(particles);

  std::string moment;
  info.at("moment").get_to(moment);

  auto it = available_moments.find(moment);
  if (it == available_moments.end())
    throw std::runtime_error(
      "Unknown moment name " + moment + " for particles " + particles);

  std::string out_dir = CONFIG().out_dir + "/" + particles + "_" + moment;

  Region region;
  region.dim = it->second > 1 ? 4 : 3;
  region.dof = it->second;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, geom_nz, region.dof);

  if (info.contains("region"))
    parse_region(info.at("region"), region, particles + " " + moment);

  if (info.contains("out_dir"))
    info.at("out_dir").get_to(out_dir);

  LOG("  {} diagnostic is added for {}, output directory: {}", moment, particles, out_dir);

  auto&& diagnostic = DistributionMoment::create( //
    out_dir, simulation_.get_named_particles(particles),
    moment_from_string(moment), region);

  if (!diagnostic)
    PetscFunctionReturn(PETSC_SUCCESS);

  diagnostics_.emplace_back(std::move(diagnostic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
