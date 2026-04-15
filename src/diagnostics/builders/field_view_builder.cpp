#include "field_view_builder.h"

#include "src/utils/configuration.h"
#include "src/utils/vector_utils.h"

FieldViewBuilder::FieldViewBuilder(
  interfaces::Simulation& simulation, std::vector<Diagnostic_up>& diagnostics)
  : DiagnosticBuilder(simulation, diagnostics)
{
}

PetscErrorCode FieldViewBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  Region region;
  region.dim = 4;
  region.dof = 3;
  region.start = Vector4I(0);
  region.size = Vector4I(geom_nx, geom_ny, geom_nz, 3);

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

  LOG("  field view diagnostic is added for {}, suffix: {}", field, suffix.empty() ? "<empty>" : suffix);

  if (!suffix.empty())
    suffix = "_" + suffix;

  /// @todo Maybe read the `res_dir` from the user instead of suffixes?
  std::string res_dir = CONFIG().out_dir + "/" + field + suffix;

  if (auto&& diagnostic = FieldView::create(res_dir, da, f, region)) {
    diagnostics_.emplace_back(std::move(diagnostic));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

void FieldViewBuilder::parse_field(const Configuration::json_t& info, DM& da,
  Vec& f, Region& region, const std::string& name)
{
  DM da_field = simulation_.da;
  DM da_rho = simulation_.da_rho;

  std::map<std::string, std::pair<DM, Vec>> map{
    {"E", {da_field, simulation_.E}},
    {"B", {da_field, simulation_.B}},
    {"J", {da_field, simulation_.J}},
    {"B0", {da_field, simulation_.B0}},
  };

  for (auto&& sort : simulation_.particles_) {
    std::string J_name = sort->parameters.sort_name + "_J";
    std::string rho_name = sort->parameters.sort_name + "_rho";
    map[J_name] = std::make_pair(da_field, sort->J);
    map[rho_name] = std::make_pair(da_rho, sort->rho[1]);
  }

  if (!map.contains(name))
    throw std::runtime_error("Incorrect name is used for " + name + ".");

  std::tie(da, f) = map.at(name);

  if (da == da_rho) {
    region.dim = 3;
    region.dof = 1;
    region.start[3] = 0;
    region.size[3] = 1;
  }
}

void FieldViewBuilder::parse_region_start_size(
  const Configuration::json_t& info, Region& region, const std::string& name)
{
  Vector3R start{0.0};
  Vector3R size{Geom};

  std::string type = "3D";

  if (auto it = info.find("type"); it != info.end())
    it->get_to(type);

  PetscInt dim = (type == "3D") ? 3 : (type == "2D") ? 2 : -1;

  if (dim < 0)
    throw std::runtime_error("Incorrect type is used for " + name + ".");

  if (info.contains("start"))
    start = parse_vector(info, "start");

  if (info.contains("size"))
    size = parse_vector(info, "size");

  if (type == "2D") {
    std::string plane;
    PetscReal position;
    parse_plane_position(info, plane, position);

    if (plane == "X") {
      start = Vector3R{position, start[Y], start[Z]};
      size = Vector3R{dx, size[Y], size[Z]};
    }
    else if (plane == "Y") {
      start = Vector3R{start[X], position, start[Z]};
      size = Vector3R{size[X], dy, size[Z]};
    }
    else if (plane == "Z") {
      start = Vector3R{start[X], start[Y], position};
      size = Vector3R{size[X], size[Y], dz};
    }
  }

  for (PetscInt i = 0; i < 3; ++i) {
    region.start[i] = FLOOR_STEP(start[i], Dx[i]);
    region.size[i] = FLOOR_STEP(size[i], Dx[i]);
  }
}

void FieldViewBuilder::parse_res_dir_suffix(
  const Configuration::json_t& info, std::string& suffix)
{
  std::string type = "3D";

  if (info.contains("type"))
    info.at("type").get_to(type);

  if (type == "2D") {
    std::string plane;
    PetscReal position;
    parse_plane_position(info, plane, position);

    Axis dir = get_component(plane);
    PetscInt position_format = FLOOR_STEP(position, Dx[dir]);
    suffix += std::format("plane{}_{:04d}", plane, position_format);
  }
}

void FieldViewBuilder::parse_plane_position(
  const Configuration::json_t& info, std::string& plane, PetscReal& position)
{
  info.at("plane").get_to(plane);

  if (plane == "X")
    position = 0.5 * geom_x;
  else if (plane == "Y")
    position = 0.5 * geom_y;
  else if (plane == "Z")
    position = 0.5 * geom_z;

  if (info.contains("position"))
    info.at("position").get_to(position);
}

void FieldViewBuilder::check_region(
  const Region& region, const std::string& name) const
{
  Vector3I start = vector_cast(region.start);
  Vector3I size = vector_cast(region.size);

  if (bool success = is_region_within_bounds(start, size, 0, Geom_n); !success) {
    throw std::runtime_error(
      "Region is not in global boundaries for " + name + " diagnostic.");
  }

  if (bool success = (size[X] > 0) && (size[Y] > 0) && (size[Z] > 0); !success)
    throw std::runtime_error("Sizes are invalid for " + name + " diagnostic.");
}
