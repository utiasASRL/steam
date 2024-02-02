#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_acc/variable.hpp"

namespace steam {
namespace traj {
namespace singer {

using Variable = steam::traj::const_acc::Variable;

inline Eigen::Matrix<double, 18, 18> getQ(
    const double& dt, const Eigen::Matrix<double, 6, 1>& add,
    const Eigen::Matrix<double, 6, 1>& qcd =
        Eigen::Matrix<double, 6, 1>::Ones()) {
  // clang-format off
  Eigen::Matrix<double, 18, 18> Q = Eigen::Matrix<double, 18, 18>::Zero();
  for (int i = 0; i < 6; ++i) {
    const double ad = add(i, 0);
    const double qc = qcd(i, 0);
    if (fabs(ad) >= 1.0) {
      const double adi = 1.0 / ad;
      const double adi2 = adi * adi;
      const double adi3 = adi * adi2;
      const double adi4 = adi * adi3;
      const double adi5 = adi * adi4;
      const double adt = ad * dt;
      const double adt2 = adt * adt;
      const double adt3 = adt2 * adt;
      const double expon = std::exp(-adt);
      const double expon2 = std::exp(-2 * adt);
      Q(i, i) = qc * (
          0.5
          * adi5
          * (1 - expon2 + 2 * adt + (2.0 / 3.0) * adt3 - 2 * adt2 - 4 * adt * expon)
      );
      Q(i, i + 6) = Q(i + 6, i) = qc * (
          0.5 * adi4 * (expon2 + 1 - 2 * expon + 2 * adt * expon - 2 * adt + adt2)
      );
      Q(i, i + 12) = Q(i + 12, i) = qc * 0.5 * adi3 * (1 - expon2 - 2 * adt * expon);
      Q(i + 6, i + 6) = qc * 0.5 * adi3 * (4 * expon - 3 - expon2 + 2 * adt);
      Q(i + 6, i + 12) = Q(i + 12, i + 6) = qc * 0.5 * adi2 * (expon2 + 1 - 2 * expon);
      Q(i + 12, i + 12) = qc * 0.5 * adi * (1 - expon2);
    } else {
      const double dt2 = dt * dt;
      const double dt3 = dt * dt2;
      const double dt4 = dt * dt3;
      const double dt5 = dt * dt4;
      const double dt6 = dt * dt5;
      const double dt7 = dt * dt6;
      const double dt8 = dt * dt7;
      const double dt9 = dt * dt8;
      const double ad2 = ad * ad;
      const double ad3 = ad * ad2;
      const double ad4 = ad * ad3;
      // use Taylor series expansion about ad = 0
      Q(i, i) = qc * (
          0.05 * dt5
          - 0.0277778 * dt6 * ad
          + 0.00992063 * dt7 * ad2
          - 0.00277778 * dt8 * ad3
          + 0.00065586 * dt9 * ad4
      );
      Q(i, i + 6) = Q(i + 6, i) = qc * (
          0.125 * dt4
          - 0.0833333 * dt5 * ad
          + 0.0347222 * dt6 * ad2
          - 0.0111111 * dt7 * ad3
          + 0.00295139 * dt8 * ad4
      );
      Q(i, i + 12) = Q(i + 12, i) = qc * (
          (1 / 6) * dt3
          - (1 / 6) * dt4 * ad
          + 0.0916667 * dt5 * ad2
          - 0.0361111 * dt6 * ad3
          + 0.0113095 * dt7 * ad4
      );
      Q(i + 6, i + 6) = qc * (
          (1 / 3) * dt3
          - 0.25 * dt4 * ad
          + 0.116667 * dt5 * ad2
          - 0.0416667 * dt6 * ad3
          + 0.0123016 * dt7 * ad4
      );
      Q(i + 6, i + 12) = Q(i + 12, i + 6) = qc * (
          0.5 * dt2
          - 0.5 * dt3 * ad
          + 0.291667 * dt4 * ad2
          - 0.125 * dt5 * ad3
          + 0.0430556 * dt6 * ad4
      );
      Q(i + 12, i + 12) = qc * (
          dt
          - dt2 * ad
          + (2 / 3) * dt3 * ad2
          - (1 / 3) * dt4 * ad3
          + 0.133333 * dt5 * ad4
      );
    }
  }
  // clang-format on

  return Q;
}

inline Eigen::Matrix<double, 18, 18> getTran(
    const double& dt, const Eigen::Matrix<double, 6, 1>& add) {
  // clang-format off
  Eigen::Matrix<double, 18, 18> Tran = Eigen::Matrix<double, 18, 18>::Identity();
  for (int i = 0; i < 6; ++i) {
    const double ad = add(i, 0);
    if (fabs(ad) >= 1.0) {
      const double adinv = 1.0 / ad;
      const double adt = ad * dt;
      const double expon = std::exp(-adt);
      Tran(i, i + 12) = (adt - 1.0 + expon) * adinv * adinv;  // C1
      Tran(i + 6, i + 12) = (1.0 - expon) * adinv;  // C2
      Tran(i + 12, i + 12) = expon;  // C3
    } else {
      const double dt2 = dt * dt;
      const double dt3 = dt * dt2;
      const double dt4 = dt * dt3;
      const double dt5 = dt * dt4;
      const double dt6 = dt * dt5;
      const double ad2 = ad * ad;
      const double ad3 = ad * ad2;
      const double ad4 = ad * ad3;
      Tran(i, i + 12) = (
          0.5 * dt2
          - (1 / 6) * dt3 * ad
          + (1 / 24) * dt4 * ad2
          - (1 / 120) * dt5 * ad3
          + (1 / 720) * dt6 * ad4
      );
      Tran(i + 6, i + 12) = (
          dt
          - 0.5 * dt2 * ad
          + (1 / 6) * dt3 * ad2
          - (1 / 24) * dt4 * ad3
          + (1 / 120) * dt5 * ad4
      );
      Tran(i + 12, i + 12) = (
          1
          - dt * ad
          + 0.5 * dt2 * ad2
          - (1 / 6) * dt3 * ad3
          + (1 / 24) * dt4 * ad4
      );
    }
  }
  Tran.block<6, 6>(0, 6).diagonal() = dt * Eigen::Array<double, 6, 1>::Ones();
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
  jacobian.block<6, 6>(6, 0) = -0.5 * lgmath::se3::curlyhat(w2) * Jinv_12;
  jacobian.block<6, 6>(12, 0) =
      -0.25 * lgmath::se3::curlyhat(w2) * lgmath::se3::curlyhat(w2) * Jinv_12 -
      0.5 * lgmath::se3::curlyhat(dw2) * Jinv_12;
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