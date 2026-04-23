#ifndef SRC_DIAGNOSTICS_FIELD_VIEW_ZAVG_H
#define SRC_DIAGNOSTICS_FIELD_VIEW_ZAVG_H

#include "src/diagnostics/field_view.h"

class FieldViewZAvg : public FieldView {
public:
  static std::unique_ptr<FieldViewZAvg> create(
    const std::string& out_dir, DM da, Vec field, const Region& region);

  PetscErrorCode finalize() override;
  PetscErrorCode diagnose(PetscInt t) override;

protected:
  FieldViewZAvg(const std::string& out_dir, DM da, Vec field, MPI_Comm newcomm);

  PetscErrorCode set_data_views(const Region& region) override;
  PetscErrorCode calculate();

  DM da_glob;
  Vec favg;
};

#endif  // SRC_DIAGNOSTICS_FIELD_VIEW_ZAVG_H
