#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_acc/pose_interpolator.hpp"
#include "steam/trajectory/const_acc/variable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace singer {

class PoseInterpolator : public steam::traj::const_acc::PoseInterpolator {
 public:
  using Ptr = std::shared_ptr<PoseInterpolator>;
  using ConstPtr = std::shared_ptr<const PoseInterpolator>;
  using Variable = steam::traj::const_acc::Variable;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2,
                        const Eigen::Matrix<double, 6, 1>& ad) {
    return std::make_shared<PoseInterpolator>(time, knot1, knot2, ad);
  }

  PoseInterpolator(const Time& time, const Variable::ConstPtr& knot1,
                   const Variable::ConstPtr& knot2,
                   const Eigen::Matrix<double, 6, 1>& ad)
      : steam::traj::const_acc::PoseInterpolator(time, knot1, knot2) {
    // Calculate time constants
    const double T = (knot2->time() - knot1->time()).seconds();
    const double tau = (time - knot1->time()).seconds();
    const double kappa = (knot2->time() - time).seconds();
    // Q and Transition matrix
    const auto Q_tau = getQ(tau, ad);
    const auto Q_T = getQ(T, ad);
    const auto Tran_kappa = getTran(kappa, ad);
    const auto Tran_tau = getTran(tau, ad);
    const auto Tran_T = getTran(T, ad);
    // Calculate interpolation values
    omega_ = Q_tau * Tran_kappa.transpose() * Q_T.inverse();
    lambda_ = Tran_tau - omega_ * Tran_T;
  }
};

}  // namespace singer
}  // namespace traj
}  // namespace steam