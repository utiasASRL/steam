#pragma once

#include <Eigen/Core>

namespace steam {
namespace traj {
namespace const_acc {

inline Eigen::Matrix<double, 18, 18> getQinv(
    const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) {
  // constants
  Eigen::Matrix<double, 6, 1> Qcinv_diag = 1.0 / Qc_diag.array();
  const double dtinv = 1.0 / dt;
  const double dtinv2 = dtinv * dtinv;
  const double dtinv3 = dtinv * dtinv2;
  const double dtinv4 = dtinv * dtinv3;
  const double dtinv5 = dtinv * dtinv4;

  // clang-format off
  Eigen::Matrix<double, 18, 18> Qinv = Eigen::Matrix<double, 18, 18>::Zero();

  Qinv.block<6, 6>(0, 0).diagonal() = 720.0 * dtinv5 * Qcinv_diag;
  Qinv.block<6, 6>(6, 6).diagonal() = 192.0 * dtinv3 * Qcinv_diag;
  Qinv.block<6, 6>(12, 12).diagonal() = 9.0 * dtinv * Qcinv_diag;

  Qinv.block<6, 6>(0, 6).diagonal() = Qinv.block<6, 6>(6, 0).diagonal() = (-360.0) * dtinv4 * Qcinv_diag;
  Qinv.block<6, 6>(0, 12).diagonal() = Qinv.block<6, 6>(12, 0).diagonal() = (60.0) * dtinv3 * Qcinv_diag;
  Qinv.block<6, 6>(6, 12).diagonal() = Qinv.block<6, 6>(12, 6).diagonal() = (-36.0) * dtinv2 * Qcinv_diag;
  // clang-format on

  return Qinv;
}

inline Eigen::Matrix<double, 3, 3> getQinv(const double& dt) {
  // constants
  const double dtinv = 1.0 / dt;
  const double dtinv2 = dtinv * dtinv;
  const double dtinv3 = dtinv * dtinv2;
  const double dtinv4 = dtinv * dtinv3;
  const double dtinv5 = dtinv * dtinv4;

  Eigen::Matrix<double, 3, 3> Qinv = Eigen::Matrix<double, 3, 3>::Zero();

  Qinv(0, 0) = 720.0 * dtinv5;
  Qinv(1, 1) = 192.0 * dtinv3;
  Qinv(2, 2) = 9.0 * dtinv;

  Qinv(0, 1) = Qinv(1, 0) = (-360.0) * dtinv4;
  Qinv(0, 2) = Qinv(2, 0) = (60.0) * dtinv3;
  Qinv(1, 2) = Qinv(2, 1) = (-36.0) * dtinv2;

  return Qinv;
}

inline Eigen::Matrix<double, 3, 3> getQ(const double& dt) {
  // constants
  const double dt2 = dt * dt;
  const double dt3 = dt * dt2;
  const double dt4 = dt * dt3;
  const double dt5 = dt * dt4;

  Eigen::Matrix<double, 3, 3> Q = Eigen::Matrix<double, 3, 3>::Zero();

  Q(0, 0) = dt5 / 20.0;
  Q(1, 1) = dt3 / 3.0;
  Q(2, 2) = dt;

  Q(0, 1) = Q(1, 0) = dt4 / 8.0;
  Q(0, 2) = Q(2, 0) = dt3 / 6.0;
  Q(1, 2) = Q(2, 1) = dt2 / 2.0;

  return Q;
}

inline Eigen::Matrix<double, 3, 3> getTran(const double& dt) {
  Eigen::Matrix<double, 3, 3> Tran = Eigen::Matrix<double, 3, 3>::Zero();
  Tran(0, 0) = Tran(1, 1) = Tran(2, 2) = 1.0;
  Tran(1, 0) = Tran(2, 0) = Tran(2, 1) = 0.0;
  Tran(0, 1) = Tran(1, 2) = dt;
  Tran(0, 2) = 0.5 * dt * dt;
  return Tran;
}

}  // namespace const_acc
}  // namespace traj
}  // namespace steam