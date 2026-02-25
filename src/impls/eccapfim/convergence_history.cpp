#include "convergence_history.h"

namespace eccapfim {

ConvergenceHistory::ConvergenceHistory(const Simulation& simulation)
  : TableDiagnostic(CONFIG().out_dir + "/temporal/convergence_history.txt"),
    simulation(simulation)
{
}

PetscErrorCode ConvergenceHistory::add_columns(PetscInt t)
{
  PetscFunctionBeginUser;
  add(6, "Time", "{:d}", t);

  SNES snes = simulation.snes;

  PetscInt nit, lit, len;
  PetscReal* hist;

  PetscCall(SNESGetIterationNumber(snes, &nit));
  PetscCall(SNESGetLinearSolveIterations(snes, &lit));
  PetscCall(SNESGetConvergenceHistory(snes, &hist, nullptr, &len));

  add(6, "NItNum", "{:d}", nit);
  add(6, "LItNum", "{:d}", lit);

  for (const auto& sort : simulation.particles_) {
    const auto& name = sort->parameters.sort_name;
    auto cn = sort->get_average_iteration_number();
    auto atc = sort->get_average_number_of_traversed_cells();
    auto mtc = sort->get_maximum_number_of_traversed_cells();
    add(8, "AvgCN_" + name, "{:.3f}", cn);
    add(8, "AvgTC_" + name, "{:.3f}", atc);
    add(8, "MaxTC_" + name, "{:3d}", mtc);
  }

  add_separator();

  if (len == 0) {
    add(12, "ConvHist", "{}", "");
  }
  else {
    for (PetscInt i = 0; i < len; ++i)
      add(12, "ConvHist", "{:8.6e}", hist[i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

}  // namespace eccapfim
