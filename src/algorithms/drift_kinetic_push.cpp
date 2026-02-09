#include "drift_kinetic_push.h"

DriftKineticPush::DriftKineticPush(PetscReal qm, PetscReal mp)
  : qm(qm), mp(mp)
{
}

void DriftKineticPush::set_tolerances(
  PetscReal atol, PetscReal rtol, PetscInt maxit)
{
  this->atol = atol;
  this->rtol = rtol;
  this->maxit = maxit;
}

void DriftKineticPush::set_qm(PetscReal qm)
{
  this->qm = qm;
}


void DriftKineticPush::set_mp(PetscReal mp)
{
  this->mp = mp;
}

PetscReal DriftKineticPush::get_mp() const
{
  return this->mp;
}

PetscReal DriftKineticPush::get_qm() const
{
  return this->qm;
}

PetscInt DriftKineticPush::get_iteration_number() const
{
  return it;
}

void DriftKineticPush::set_fields_callback(SetFields&& callback)
{
  this->set_fields = std::move(callback);
}


void DriftKineticPush::process(
  PetscReal dt, PointByField& pn, const PointByField& p0)
{
  pre_step(dt, pn, p0);

  PetscAssertAbort((bool)set_fields, PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::set_fields have to be specified");

  for (it = 0; it < maxit; ++it) {
    update_Vp(pn, p0);

    if (check_discrepancy(dt, pn, p0) && it) {
      //update_v_perp(pn, p0);
      return;
    }

    update_r(dt, pn, p0);
    update_fields(pn, p0);
    update_v_parallel(dt, pn, p0);
  }

  PetscCheckAbort((Rn >= atol + rtol * R0) || (Vn >= atol + rtol * V0), PETSC_COMM_WORLD, PETSC_ERR_USER,
    "DriftKineticPush::process() nonlinear iterations diverged with norm %e and %e!", Rn, Vn);
}

void DriftKineticPush::pre_step(const PetscReal dt, PointByField& pn, const PointByField& p0) {
  update_fields(pn, p0);
  Vh = pn.p_parallel;
  Vp = Vh * bh;
  R0 = get_residue_r(dt, pn, p0);
  V0 = get_residue_v(dt, pn, p0);
  Rn = 0.0;
  Vn = 0.0;
}

void DriftKineticPush::update_Vp(const PointByField& pn, const PointByField& p0) {
  Vh = 0.5 * (pn.p_parallel + p0.p_parallel);
  Vp = Vh * bh + get_Vd(p0);
}

Vector3R DriftKineticPush::get_Vd(const PointByField& p0) const
{
  return (Vh * Vh / lenBh + p0.mu_p / mp) * bh.cross(gradBh / lenBh) / qm + Eh.cross(bh) / lenBh;
}

bool DriftKineticPush::check_discrepancy(PetscReal dt, const PointByField& pn, const PointByField& p0){
  Rn = get_residue_r(dt, pn, p0);
  Vn = get_residue_v(dt, pn, p0);
  return (Rn < atol + rtol * R0) && (Vn < atol + rtol * V0);
}

PetscReal DriftKineticPush::get_residue_r(PetscReal dt, const PointByField& pn,
  const PointByField& p0) const
{
  //return (pn.r - p0.r - dt * Vp).length()/pn.r.length();
  return (pn.r - p0.r - dt * Vp).length();
}

PetscReal DriftKineticPush::get_residue_v(PetscReal dt, const PointByField& pn,
  const PointByField& p0) const
{
  //return std::abs((pn.p_parallel - p0.p_parallel) - dt * get_v_parallel(p0))/std::abs(pn.p_parallel);
  return std::abs((pn.p_parallel - p0.p_parallel) - dt * get_v_parallel(p0));
}

void DriftKineticPush::update_r(PetscReal dt, PointByField& pn, const PointByField& p0) const
{
  pn.r = p0.r + dt * Vp;
}

void DriftKineticPush::update_v_perp(PointByField& pn, const PointByField& p0)
{
  Vector3R Edummy, gradBdummy;
  set_fields(p0.r, pn.r, Edummy, Bn, gradBdummy);
  pn.p_perp = std::sqrt(2.*pn.mu_p*Bn.length() / mp);
}

void DriftKineticPush::update_v_parallel(PetscReal dt, PointByField& pn, const PointByField& p0) const
{
  pn.p_parallel = p0.p_parallel + dt * get_v_parallel(p0) + get_F(p0) / mp;
}

PetscReal DriftKineticPush::get_F(const PointByField& p0) const {
  PetscReal F = (std::abs(Vh) < 1e-12) ?  0. : - p0.mu_p * (bh - b0).dot(meanB) / Vh;
  LOG("F = {}", F);
  return F;
}

PetscReal DriftKineticPush::get_v_parallel(const PointByField& p0) const {
  PetscReal qm_term = (std::abs(Vh) < 1e-12) ? Eh.dot(bh) : (Eh.dot(Vp) / Vh);
  PetscReal mu_term = (std::abs(Vh) < 1e-12) ? gradBh.dot(bh) : (gradBh.dot(Vp) / Vh);
  return qm * qm_term - (p0.mu_p / mp) * mu_term;
}

void DriftKineticPush::update_fields(const PointByField& pn, const PointByField& p0) {
  Vector3R Edummy, gradBdummy;
  #if 1
  set_fields(p0.r, p0.r, Edummy, B0, gradBdummy);
  set_fields(p0.r, pn.r, Eh, Bh, gradBh);
  #endif
  meanB = 0.5 * (Bh + B0);
  bh = Bh.normalized(), b0 = B0.normalized();
  lenBh = Bh.length();
  #if 0
  LOG("update_fields:");
  LOG("it = {}", it);
  LOG("Rn = {}, Vn = {}", Rn, Vn);
  LOG("p0.x = {}, p0.y = {}, p0.z = {}", p0.r.x(), p0.r.y(), p0.r.z());
  LOG("pn.x = {}, pn.y = {}, pn.z = {}", pn.r.x(), pn.r.y(), pn.r.z());
  LOG("Eh.x = {}, Eh.y = {}, Eh.z = {}", Eh.x(), Eh.y(), Eh.z());
  LOG("Bh.x = {}, Bh.y = {}, Bh.z = {}", Bh.x(), Bh.y(), Bh.z());
  LOG("gradBh.x = {}, gradBh.y = {}, gradBh.z = {}", gradBh.x(), gradBh.y(), gradBh.z());
  LOG("lenBh = {}", lenBh);
  LOG("Vp = ({}, {}, {})", Vp.x(), Vp.y(), Vp.z());
  #endif
}
