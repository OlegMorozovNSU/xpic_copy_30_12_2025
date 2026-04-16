#include "table_diagnostic.h"

TableDiagnostic::TableDiagnostic(const std::string& filename)
{
  PetscCallAbort(PETSC_COMM_WORLD, open(filename));
}

PetscErrorCode TableDiagnostic::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t == 0)
    PetscCall(initialize());

  PetscCall(add_columns(t));

  if (!values.empty()) {
    if (t == 0)
      PetscCall(write_formatted(titles));
    PetscCall(write_formatted(values));

    titles.clear();
    values.clear();
  }

  if (t % diagnose_period_ == 0)
    PetscCall(flush());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::initialize()
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::add_columns(PetscInt /* t */)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode TableDiagnostic::write_formatted(
  const std::vector<std::string>& container)
{
  PetscFunctionBeginUser;
  PetscInt i = 0, size = (PetscInt)container.size();

  for (; i < size - 1; ++i) {
    file << container[i] << "  ";
  }

  auto last = container.back();
  while (!last.empty() && last.back() == ' ') {
    last.pop_back();
  }

  file << last << "\n";
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::open(const std::string& filename)
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  PetscCall(close());

  std::filesystem::path path(filename);
  std::filesystem::create_directories(path.parent_path());

  PetscCallCXX(file.open(path, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::flush()
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  PetscCallCXX(file.flush());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TableDiagnostic::close()
{
  if (!is_synchronized())
    return PETSC_SUCCESS;

  PetscFunctionBeginHot;
  if (file.is_open()) {
    PetscCallCXX(file.flush());
    PetscCallCXX(file.close());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

bool TableDiagnostic::is_synchronized()
{
  PetscMPIInt flag;
  PetscCallMPI(MPI_Initialized(&flag));
  if (!static_cast<bool>(flag))
    return true;

  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0)
    return true;

  return false;
}
