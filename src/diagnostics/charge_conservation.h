#ifndef SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
#define SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H

#include "src/interfaces/simulation.h"
#include "src/diagnostics/utils/table_diagnostic.h"

class ChargeConservation : public TableDiagnostic {
public:
  ChargeConservation(const interfaces::Simulation& simulation);

private:
  PetscErrorCode add_columns(PetscInt t) override;

  const interfaces::Simulation& simulation;
};

#endif  // SRC_DIAGNOSTICS_CHARGE_CONSERVATION_H
