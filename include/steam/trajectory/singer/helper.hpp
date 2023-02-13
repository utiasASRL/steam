#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_acc/variable.hpp"

namespace steam {
namespace traj {
namespace singer {

using Variable = steam::traj::const_acc::Variable;

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
  Eigen::Matrix<double, 18, 18> Tran = Eigen::Matrix<double, 18, 18>::Identity();
  Tran.block<6, 6>(0, 6).diagonal() = dt * Eigen::Array<double, 6, 1>::Ones();
  Tran.block<6, 6>(0, 12).diagonal() = (adt - 1 + expon) * adinv2;
  Tran.block<6, 6>(6, 12).diagonal() = (1 - expon) * adinv;
  Tran.block<6, 6>(12, 12).diagonal() = expon;
  // clang-format on
  return Tran;
}

// See State Estimation (2nd Ed) Section 11.1.4 and page 66 of Tim Tang's thesis
// "F" in SE book
inline Eigen::Matrix<double, 18, 18> getJacKnot1(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2,
    const Eigen::Matrix<double, 6, 1>& ad) {
  // precompute
  const auto T_21 = knot2->pose()->value() / knot1->pose()->value();
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const double dt = (knot2->time() - knot1->time()).seconds();
  const Eigen::Matrix<double, 18, 18> Phi = getTran(dt, ad);
  const auto Jinv_12 = J_21_inv * T_21.adjoint();
  const auto w2 = knot2->velocity()->value();
  const auto dw2 = knot2->acceleration()->value();
  // init jacobian
  Eigen::Matrix<double, 18, 18> jacobian;
  jacobian.setZero();
  // pose
  jacobian.block<6, 6>(0, 0) = -Jinv_12;
  jacobian.block<6, 6>(6, 0) =
    -0.5 * lgmath::se3::curlyhat(w2) * Jinv_12;
  jacobian.block<6, 6>(12, 0) =
    -0.25 * lgmath::se3::curlyhat(w2) * lgmath::se3::curlyhat(w2) * Jinv_12
    - 0.5 * lgmath::se3::curlyhat(dw2) * Jinv_12;
  // velocity
  jacobian.block<6, 6>(0, 6) = -Phi.block<6, 6>(0, 6);
  jacobian.block<6, 6>(6, 6) = -Phi.block<6, 6>(6, 6);
  jacobian.block<6, 6>(12, 6) = Eigen::Matrix<double, 6, 6>::Zero();
  // acceleration
  jacobian.block<6, 6>(0, 12) = -Phi.block<6, 6>(0, 12);
  jacobian.block<6, 6>(6, 12) = -Phi.block<6, 6>(6, 12);
  jacobian.block<6, 6>(12, 12) = -Phi.block<6, 6>(12, 12);
  return jacobian;
}

// Note: getJacKnot2 is the same for WNOJ and Singer, no need to redefine.

}  // namespace singer
}  // namespace traj
}  // namespace steam