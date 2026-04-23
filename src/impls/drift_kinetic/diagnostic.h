#ifndef SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H
#define SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H

#include "src/diagnostics/distribution_moment.h"
#include "src/diagnostics/table_diagnostic.h"
#include "src/impls/drift_kinetic/particles.h"
#include "src/utils/configuration.h"

namespace drift_kinetic {

class Simulation;

class DkDistributionMoment : public ::DistributionMoment {
public:
  static std::unique_ptr<DkDistributionMoment> create(const std::string& out_dir,
    const Particles& particles, const Moment& moment, const Region& region);

protected:
  DkDistributionMoment(const std::string& out_dir,
    const Particles& particles, const Moment& moment, MPI_Comm newcomm);

  PetscErrorCode collect() override;

  const Particles& dk_particles;
};

class PointByFieldTrace : public TableDiagnostic {
public:
  PointByFieldTrace(const std::string& out_dir, const Particles& particles, PetscInt skip = 1);


  PetscErrorCode initialize() override;
  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

private:
  PetscInt skip;
  const Particles& particles;

  PetscErrorCode add_columns(PetscInt t) override;
};

class EnergyConservation : public TableDiagnostic {
public:
  EnergyConservation(const Simulation& simulation);
  PetscErrorCode diagnose(PetscInt t) override;
  PetscErrorCode initialize() override;
  PetscErrorCode finalize() override;
  PetscErrorCode add_columns(PetscInt t) override;

  const Simulation& simulation;
  PetscReal w_E = 0, w_E0 = 0;
  PetscReal w_B = 0, w_B0 = 0;
  PetscReal dF = 0;
  PetscReal a_EJ = 0;
  PetscReal a_MB = 0, a_MB0 = 0;
  PetscReal w_M = 0, w_Mn = 0;
  PetscReal K0 = 0, K = 0;
  bool initialized = false;

private:
  PetscErrorCode init_charge_conservation();
  PetscErrorCode collect_charge_density(PetscInt sort_id);
  PetscErrorCode collect_charge_densities();
  void calculate_kinetic_energies(std::vector<PetscReal>& per_sort, PetscReal& total) const;

  DM charge_da = nullptr;
  Mat divE = nullptr;
  std::vector<PetscReal> K0_by_sort;
  std::vector<PetscReal> K_by_sort;
  std::vector<Vec> charge_locals;
  std::vector<Vec> charge_fields;
  std::vector<Vec> current_densities;
};

} // namespace drift_kinetic

#endif // SRC_IMPLS_DRIFT_KINETIC_DIAGNOSTIC_H
