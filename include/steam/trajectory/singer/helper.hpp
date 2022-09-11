#pragma once

#include <Eigen/Core>

namespace steam {
namespace traj {
namespace singer {

inline Eigen::Matrix<double, 18, 18> getQ(
    const double& dt, const Eigen::Matrix<double, 6, 1>& ad,
    const Eigen::Matrix<double, 6, 1>& qcd =
        Eigen::Matrix<double, 6, 1>::Ones()) {
  // constants
  Eigen::Array<double, 6, 1> adinv = 1.0 / ad.array();
  Eigen::Array<double, 6, 1> adinv2 = adinv * adinv;
  Eigen::Array<double, 6, 1> adinv3 = adinv * adinv2;
  Eigen::Array<double, 6, 1> adinv4 = adinv * adinv3;

  Eigen::Array<double, 6, 1> adt = ad * dt;
  Eigen::Array<double, 6, 1> adt2 = adt * adt;
  Eigen::Array<double, 6, 1> adt3 = adt2 * adt;

  Eigen::Array<double, 6, 1> expon = (-adt).exp();
  Eigen::Array<double, 6, 1> expon2 = (-2 * adt).exp();

  // clang-format off
  Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Zero();
  Q.block<6, 6>(0, 0).diagonal() = qcd.array() * adinv4 * (1 - expon2 + 2 * adt + 2. / 3. * adt3 - 2 * adt2 - 4 * adt * expon);
  Q.block<6, 6>(0, 6).diagonal() = Q.block<6, 6>(6, 0).diagonal() = qcd.array() * adinv3 * (expon2 + 1 - 2 * expon + 2 * adt * expon - 2 * adt + adt2);
  Q.block<6, 6>(0, 12).diagonal() = Q.block<6, 6>(12, 0).diagonal() = qcd.array() * adinv2 * (1 - expon2 - 2 * adt * expon);
  Q.block<6, 6>(6, 6).diagonal() = qcd.array() * adinv2 * (4 * expon - 3 - expon2 + 2 * adt);
  Q.block<6, 6>(6, 12).diagonal() = Q.block<6, 6>(12, 6).diagonal() = qcd.array() * adinv * (expon2 + 1 - 2 * expon);
  Q.block<6, 6>(12, 12).diagonal() = qcd.array() * (1 - expon2);
  // clang-format on

  return Q;
}

inline Eigen::Matrix<double, 18, 18> getTran(
    const double& dt, const Eigen::Matrix<double, 6, 1>& ad) {
  Eigen::Array<double, 6, 1> adinv = 1.0 / ad.array();
  Eigen::Array<double, 6, 1> adinv2 = adinv * adinv;
  Eigen::Array<double, 6, 1> adt = ad.array() * dt;
  Eigen::Array<double, 6, 1> expon = (-adt).exp();

  // clang-format off
  Eigen::Matrix<double, 18, 18> Tran = Eigen::Matrix<double, 18, 18>::Zero();
  Tran.block<6, 6>(0, 0).diagonal() = Tran.block<6, 6>(6, 6).diagonal() = Eigen::Array<double, 6, 1>::Ones();
  Tran.block<6, 6>(6, 0).diagonal() = Tran.block<6, 6>(12, 0).diagonal() = Tran.block<6, 6>(12, 6).diagonal() = Eigen::Array<double, 6, 1>::Zero();
  Tran.block<6, 6>(0, 6).diagonal() = dt * Eigen::Array<double, 6, 1>::Ones();
  Tran.block<6, 6>(0, 12).diagonal() = (adt - 1 + expon) * adinv2;
  Tran.block<6, 6>(6, 12).diagonal() = (1 - expon) * adinv;
  Tran.block<6, 6>(12, 12).diagonal() = expon;
  // clang-format on

  return Tran;
}

}  // namespace singer
}  // namespace traj
}  // namespace steam