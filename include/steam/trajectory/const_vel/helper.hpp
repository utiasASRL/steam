#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_vel/variable.hpp"

namespace steam {
namespace traj {
namespace const_vel {

inline Eigen::Matrix<double, 12, 12> getJacKnot1(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
  // precompute
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  double dt = (knot2->time() - knot1->time()).seconds();
  const auto Jinv_12 = J_21_inv * T_21.adjoint();

  // init jacobian
  Eigen::Matrix<double, 12, 12> jacobian;
  jacobian.setZero();

  // pose
  jacobian.block<6, 6>(0, 0) = -Jinv_12;
  jacobian.block<6, 6>(6, 0) =
      -0.5 * lgmath::se3::curlyhat(knot2->velocity()->value()) * Jinv_12;

  // velocity
  jacobian.block<6, 6>(0, 6) = -dt * Eigen::Matrix<double, 6, 6>::Identity();
  jacobian.block<6, 6>(6, 6) = -Eigen::Matrix<double, 6, 6>::Identity();

  return jacobian;
}

inline Eigen::Matrix<double, 12, 12> getJacKnot2(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
  // precompute
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  // double deltaTime = (knot2->time() - knot1->time()).seconds();

  // init jacobian
  Eigen::Matrix<double, 12, 12> jacobian;
  jacobian.setZero();

  // pose
  jacobian.block<6, 6>(0, 0) = J_21_inv;
  jacobian.block<6, 6>(6, 0) =
      0.5 * lgmath::se3::curlyhat(knot2->velocity()->value()) * J_21_inv;

  // velocity
  jacobian.block<6, 6>(6, 6) = J_21_inv;

  return jacobian;
}

// inverse of getJacKnot2
inline Eigen::Matrix<double, 12, 12> getJacKnot3(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
  // precompute
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  const auto xi_21 = T_21.vec();
  const auto J_21 = lgmath::se3::vec2jac(xi_21);

  // init jacobian
  Eigen::Matrix<double, 12, 12> gamma_inv;
  gamma_inv.setZero();

  // pose
  gamma_inv.block<6, 6>(0, 0) = J_21;
  gamma_inv.block<6, 6>(6, 0) =
      -0.5 * J_21 * lgmath::se3::curlyhat(knot2->velocity()->value());

  // velocity
  gamma_inv.block<6, 6>(6, 6) = J_21;

  return gamma_inv;
}

inline Eigen::Matrix<double, 12, 12> getXi(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
//   const auto Tau_21 = lgmath::se3::tranAd(T_21);
  const auto Tau_21 = T_21.adjoint();
  Eigen::Matrix<double, 12, 12> Xi;
  Xi.setZero();
  Xi.block<6, 6>(0, 0) = Tau_21;
  return Xi;
}

inline Eigen::Matrix<double, 12, 12> getQinv(
    const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) {
  // constants
  Eigen::Matrix<double, 6, 1> Qcinv_diag = 1.0 / Qc_diag.array();
  const double dtinv = 1.0 / dt;
  const double dtinv2 = dtinv * dtinv;
  const double dtinv3 = dtinv * dtinv2;

  // clang-format off
  Eigen::Matrix<double, 12, 12> Qinv = Eigen::Matrix<double, 12, 12>::Zero();

  Qinv.block<6, 6>(0, 0).diagonal() = 12.0 * dtinv3 * Qcinv_diag;
  Qinv.block<6, 6>(6, 6).diagonal() = 4.0 * dtinv * Qcinv_diag;
  Qinv.block<6, 6>(0, 6).diagonal() = Qinv.block<6, 6>(6, 0).diagonal() = (-6.0) * dtinv2 * Qcinv_diag;
  // clang-format on

  return Qinv;
}

inline Eigen::Matrix<double, 12, 12> getQ(
    const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) {
  // constants
  const double dt2 = dt * dt;
  const double dt3 = dt * dt2;
  // clang-format off
  Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Zero();
  Q.block<6, 6>(0, 0).diagonal() = dt3 * Qc_diag / 3.0;
  Q.block<6, 6>(6, 6).diagonal() = dt * Qc_diag;
  Q.block<6, 6>(0, 6).diagonal() = Q.block<6, 6>(6, 0).diagonal() = dt2 * Qc_diag / 2.0;
  // clang-format on
  return Q;
}

inline Eigen::Matrix<double, 12, 12> getPhi(const double& dt) {
  Eigen::Matrix<double, 12, 12> Tran = Eigen::Matrix<double, 12, 12>::Identity();
  Tran.block<6, 6>(0, 6) = Eigen::Matrix<double, 6, 6>::Identity() * dt;
  return Tran;
}

inline Eigen::Matrix<double, 2, 2> getQinv(const double& dt) {
  // constants
  const double dtinv = 1.0 / dt;
  const double dtinv2 = dtinv * dtinv;
  const double dtinv3 = dtinv * dtinv2;

  Eigen::Matrix<double, 2, 2> Qinv = Eigen::Matrix<double, 2, 2>::Zero();

  Qinv(0, 0) = 12.0 * dtinv3;
  Qinv(1, 1) = 4.0 * dtinv;
  Qinv(0, 1) = Qinv(1, 0) = (-6.0) * dtinv2;

  return Qinv;
}

inline Eigen::Matrix<double, 2, 2> getQ(const double& dt) {
  // constants
  const double dt2 = dt * dt;
  const double dt3 = dt * dt2;

  Eigen::Matrix<double, 2, 2> Q = Eigen::Matrix<double, 2, 2>::Zero();

  Q(0, 0) = dt3 / 3.0;
  Q(1, 1) = dt;
  Q(0, 1) = Q(1, 0) = dt2 / 2.0;

  return Q;
}

inline Eigen::Matrix<double, 2, 2> getTran(const double& dt) {
  Eigen::Matrix<double, 2, 2> Tran = Eigen::Matrix<double, 2, 2>::Zero();
  Tran(0, 0) = Tran(1, 1) = 1.0;
  Tran(1, 0) = 0.0;
  Tran(0, 1) = dt;
  return Tran;
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam