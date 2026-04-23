#include "field_view_zavg.h"

#include "src/utils/geometries.h"
#include "src/utils/vector_utils.h"


std::unique_ptr<FieldViewZAvg> FieldViewZAvg::create(
  const std::string& out_dir, DM da, Vec field, const Region& region)
{
  PetscFunctionBeginUser;
  MPI_Comm newcomm;
  PetscCallAbort(PETSC_COMM_WORLD, World::create_local_comm(da, region, &newcomm));

  if (newcomm == MPI_COMM_NULL)
    PetscFunctionReturn(nullptr);

  auto* diagnostic = new FieldViewZAvg(out_dir, da, field, newcomm);
  PetscCallAbort(PETSC_COMM_WORLD, diagnostic->set_data_views(region));
  PetscFunctionReturn(std::unique_ptr<FieldViewZAvg>(diagnostic));
}

FieldViewZAvg::FieldViewZAvg(
  const std::string& out_dir, DM da, Vec field, MPI_Comm newcomm)
  : FieldView(out_dir, da, field, newcomm)
{
}

/// @todo Unify this logic with `DistributionMoment` diagnostic
PetscErrorCode FieldViewZAvg::set_data_views(const Region& reg)
{
  PetscFunctionBeginUser;
  da_glob = da;
  PetscCall(World::create_local_dm(da_glob, reg, comm, &da));
  PetscCall(DMCreateGlobalVector(da, &favg));
  PetscCall(FieldView::set_data_views(reg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldViewZAvg::finalize()
{
  PetscFunctionBeginUser;
  PetscCall(FieldView::finalize());
  PetscCall(VecDestroy(&favg));
  PetscCall(DMDestroy(&da));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldViewZAvg::diagnose(PetscInt t)
{
  PetscFunctionBeginUser;
  if (t % diagnose_period_ == 0) {
    PetscCall(calculate());
    std::swap(field, favg);
    PetscCall(FieldView::diagnose(t));
    std::swap(field, favg);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FieldViewZAvg::calculate()
{
  PetscFunctionBeginUser;
  PetscCall(VecSet(favg, 0));

  PetscReal**** arr;
  PetscReal**** avg_arr;
  World* world;

  PetscCall(DMDAVecGetArrayDOFRead(da_glob, field, &arr));
  PetscCall(DMDAVecGetArrayDOFWrite(da, favg, &avg_arr));
  PetscCall(DMGetApplicationContext(da_glob, &world));

  Vector3I gstart = vector_cast(region.start);
  Vector3I gsize = vector_cast(region.size);
  gstart[Z] = world->start[Z];
  gsize[Z] = world->size[Z];

#pragma omp parallel for simd
  for (PetscInt g = 0; g < world->size.elements_product(); ++g) {
    Vector3I vg{
      world->start[X] + g % world->size[X],
      world->start[Y] + (g / world->size[X]) % world->size[Y],
      world->start[Z] + (g / world->size[X]) / world->size[Y],
    };

    if (!is_point_within_bounds(vg, gstart, gsize))
      continue;

    for (PetscInt c = 0; c < 3; c++) {
#pragma omp atomic update
      avg_arr[0][vg[Y]][vg[X]][c] +=
        arr[vg[Z]][vg[Y]][vg[X]][c] / (PetscReal)geom_nz;
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(da_glob, field, &arr));
  PetscCall(DMDAVecRestoreArrayDOFWrite(da, favg, &avg_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
