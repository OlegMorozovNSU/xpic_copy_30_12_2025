#include "set_magnetic_field_builder.h"

#include "src/commands/set_magnetic_field.h"

SetMagneticFieldBuilder::SetMagneticFieldBuilder(
  interfaces::Simulation& simulation, std::vector<Command_up>& result)
  : CommandBuilder(simulation, result)
{
}

PetscErrorCode SetMagneticFieldBuilder::build(const Configuration::json_t& info)
{
  PetscFunctionBeginUser;
  std::set<std::string_view> available_setters{
    "SetUniformField",
    "SetCoilsField",
    "SetCosineField",
    "SetGeneralCosineField",
  };

  std::string command;
  info.at("command").get_to(command);

  std::string field;
  if (command == "SetElectricField") {
    field = "E";
    if (auto it = info.find("field"); it != info.end()) {
      field = it->get<std::string>();
      if (field != "E") {
        throw std::runtime_error(
          "SetElectricField allows only field=\"E\", got: " + field);
      }
    }
  }
  else {
    info.at("field").get_to(field);
  }

  Vec field_vector = simulation_.get_named_vector(field);

  Vec field_axpy = nullptr;
  if (auto it = info.find("field_axpy"); it != info.end())
    field_axpy = simulation_.get_named_vector(it->get<std::string>());

  const Configuration::json_t& setter = info.at("setter");

  std::string name;
  setter.at("name").get_to(name);

  if (!available_setters.contains(name))
    throw std::runtime_error("Unknown setter name " + name);

  SetMagneticField::Setter setup;

  if (name == "SetUniformField") {
    LOG("  Using SetUniformField setter");
    Vector3R value = parse_vector(setter, "value");
    setup = SetUniformField(value);
    LOG("    Field value: {} {} {}", REP3_A(value));
  }
  else if (name == "SetCoilsField") {
    LOG("  Using SetCoilsField setter");
    std::vector<SetCoilsField::Coil> coils;
    for (auto& coil_info : setter.at("coils")) {
      SetCoilsField::Coil coil{
        coil_info.at("z0").get<PetscReal>(),
        coil_info.at("R").get<PetscReal>(),
        coil_info.at("I").get<PetscReal>(),
      };
      LOG("    Adding magnetic coil, z0: {}, R: {}, I: {}", coil.z0, coil.R, coil.I);
      coils.emplace_back(std::move(coil));
    }
    setup = SetCoilsField(std::move(coils));
  }
  else if (name == "SetCosineField") {
    LOG("  Using SetCosineField setter");
    BoxGeometry box;
    load_geometry(setter, box);

    Vector3R amplitude = parse_vector(setter, "amplitude");
    Vector3R wave_number = parse_vector(setter, "wave_number");

    LOG("    Cosine amplitude: {} {} {}", REP3_A(amplitude));
    LOG("    Cosine wave_number: {} {} {}", REP3_A(wave_number));
    LOG("    Cosine region min: {} {} {}", REP3_A(box.min));
    LOG("    Cosine region max: {} {} {}", REP3_A(box.max));

    setup = SetCosineField(std::move(box), amplitude, wave_number);
  }
  else if (name == "SetGeneralCosineField") {
    LOG("  Using SetGeneralCosineField setter");
    BoxGeometry box;
    load_geometry(setter, box);

    Vector3R amplitude = parse_vector(setter, "amplitude");
    Vector3R wave_number = parse_vector(setter, "wave_number");

    LOG("    Cosine amplitude: {} {} {}", REP3_A(amplitude));
    LOG("    Cosine wave_number: {} {} {}", REP3_A(wave_number));
    LOG("    Cosine region min: {} {} {}", REP3_A(box.min));
    LOG("    Cosine region max: {} {} {}", REP3_A(box.max));

    setup = SetGeneralCosineField(std::move(box), amplitude, wave_number);
  }

  commands_.emplace_back(
    std::make_unique<SetMagneticField>(field_vector, field_axpy, std::move(setup)));

  LOG("  {} command is added for {}", command, field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
