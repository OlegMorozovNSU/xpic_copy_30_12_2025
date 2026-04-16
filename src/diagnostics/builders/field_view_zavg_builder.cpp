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
  std::string field;
  info.at("field").get_to(field);

  std::string out_dir = field + "_zavg";

  DM da;
  Vec f;
  Region region;
  region.dim = 4;
  region.dof = 3;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, 1, 3);

  parse_field(info, da, f, region, field);

  if (info.contains("region"))
    parse_region(info.at("region"), region, field);

  if (info.contains("out_dir"))
    info.at("out_dir").get_to(out_dir);

  LOG("  field view (zavg) diagnostic is added for {}, output directory: {}", field, out_dir);

  if (auto&& diagnostic =
        FieldViewZAvg::create(CONFIG().out_dir + "/" + out_dir, da, f, region)) {
    diagnostics_.emplace_back(std::move(diagnostic));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
