#include "field_view_zavg_builder.h"

#include "src/diagnostics/field_view_zavg.h"

FieldViewZAvgBuilder::FieldViewZAvgBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : FieldViewBuilder(simulation, diagnostics)
{
}

PetscErrorCode FieldViewZAvgBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  Region region;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, 1, 3);
  region.dim = 4;
  region.dof = 3;

  std::string field;
  info.at("field").get_to(field);

  DM da;
  Vec f;
  parse_field(info, da, f, region, field);

  std::string suffix;

  if (auto it = info.find("region"); it != info.end()) {
    parse_region_start_size(*it, region, field);
    parse_res_dir_suffix(*it, suffix);
    check_region(region, field);
  }

  LOG("  field view (zavg) diagnostic is added for {}, suffix: {}", field, suffix.empty() ? "<empty>" : suffix);

  if (!suffix.empty())
    suffix = "_" + suffix;

  std::string res_dir = CONFIG().out_dir + "/" + field + "_zavg" + suffix;

  if (auto&& diagnostic = FieldViewZAvg::create(res_dir, da, f, region)) {
    diagnostics_.emplace_back(std::move(diagnostic));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
