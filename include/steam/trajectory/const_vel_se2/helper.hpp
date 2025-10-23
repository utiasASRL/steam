#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_vel_se2/variable.hpp"

namespace steam {
namespace traj {
namespace const_vel_se2 {

// See State Estimation (2nd Ed) Section 11.1.4 for explanation of these
// Jacobians "F" in SE book
inline Eigen::Matrix<double, 6, 6> getJacKnot1(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
  // precompute
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se2::vec2jacinv(xi_21);
  double dt = (knot2->time() - knot1->time()).seconds();
  const auto Jinv_12 = J_21_inv * T_21.adjoint();
  // init jacobian
  Eigen::Matrix<double, 6, 6> jacobian;
  jacobian.setZero();
  // pose
  jacobian.block<3, 3>(0, 0) = -Jinv_12;
  jacobian.block<3, 3>(3, 0) =
      -0.5 * lgmath::se2::curlyhat(knot2->velocity()->value()) * Jinv_12;
  // velocity
  jacobian.block<3, 3>(0, 3) = -dt * Eigen::Matrix<double, 3, 3>::Identity();
  jacobian.block<3, 3>(3, 3) = -Eigen::Matrix<double, 3, 3>::Identity();
  return jacobian;
}

// "E" in SE book
inline Eigen::Matrix<double, 6, 6> getJacKnot2(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
  // precompute
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se2::vec2jacinv(xi_21);
  // init jacobian
  Eigen::Matrix<double, 6, 6> jacobian;
  jacobian.setZero();
  // pose
  jacobian.block<3, 3>(0, 0) = J_21_inv;
  jacobian.block<3, 3>(3, 0) =
      0.5 * lgmath::se2::curlyhat(knot2->velocity()->value()) * J_21_inv;
  // velocity
  jacobian.block<3, 3>(3, 3) = J_21_inv;
  return jacobian;
}

// inverse of getJacKnot2
inline Eigen::Matrix<double, 6, 6> getJacKnot3(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
  // precompute
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  const auto xi_21 = T_21.vec();
  const auto J_21 = lgmath::se2::vec2jac(xi_21);
  // init jacobian
  Eigen::Matrix<double, 6, 6> gamma_inv;
  gamma_inv.setZero();
  // pose
  gamma_inv.block<3, 3>(0, 0) = J_21;
  gamma_inv.block<3, 3>(3, 0) =
      -0.5 * J_21 * lgmath::se2::curlyhat(knot2->velocity()->value());
  // velocity
  gamma_inv.block<3, 3>(3, 3) = J_21;
  return gamma_inv;
}

inline Eigen::Matrix<double, 6, 6> getXi(const Variable::ConstPtr& knot1,
                                           const Variable::ConstPtr& knot2) {
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  //   const auto Tau_21 = lgmath::se2::tranAd(T_21);
  const auto Tau_21 = T_21.adjoint();
  Eigen::Matrix<double, 6, 6> Xi;
  Xi.setZero();
  Xi.block<3, 3>(0, 0) = Tau_21;
  return Xi;
}

inline Eigen::Matrix<double, 6, 6> getQinv(
    const double& dt, const Eigen::Matrix<double, 3, 1>& Qc_diag) {
  // constants
  Eigen::Matrix<double, 3, 1> Qcinv_diag = 1.0 / Qc_diag.array();
  const double dtinv = 1.0 / dt;
  const double dtinv2 = dtinv * dtinv;
  const double dtinv3 = dtinv * dtinv2;
  // clang-format off
  Eigen::Matrix<double, 6, 6> Qinv = Eigen::Matrix<double, 6, 6>::Zero();
  Qinv.block<3, 3>(0, 0).diagonal() = 12.0 * dtinv3 * Qcinv_diag;
  Qinv.block<3, 3>(3, 3).diagonal() = 4.0 * dtinv * Qcinv_diag;
  Qinv.block<3, 3>(0, 3).diagonal() = Qinv.block<3, 3>(3, 0).diagonal() = (-6.0) * dtinv2 * Qcinv_diag;
  // clang-format on
  return Qinv;
}

inline Eigen::Matrix<double, 6, 6> getQ(
    const double& dt, const Eigen::Matrix<double, 3, 1>& Qc_diag) {
  // constants
  const double dt2 = dt * dt;
  const double dt3 = dt * dt2;
  // clang-format off
  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  Q.block<3, 3>(0, 0).diagonal() = dt3 * Qc_diag / 3.0;
  Q.block<3, 3>(3, 3).diagonal() = dt * Qc_diag;
  Q.block<3, 3>(0, 3).diagonal() = Q.block<3, 3>(3, 0).diagonal() = dt2 * Qc_diag / 2.0;
  // clang-format on
  return Q;
}

inline Eigen::Matrix<double, 6, 6> getTran(const double& dt) {
  Eigen::Matrix<double, 6, 6> Tran =
      Eigen::Matrix<double, 6, 6>::Identity();
  Tran.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity() * dt;
  return Tran;
}

}  // namespace const_vel_se2
}  // namespace traj
}  // namespace steam