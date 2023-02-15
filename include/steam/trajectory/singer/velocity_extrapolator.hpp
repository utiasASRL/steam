#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_acc/variable.hpp"
#include "steam/trajectory/const_acc/velocity_extrapolator.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace singer {

class VelocityExtrapolator
    : public steam::traj::const_acc::VelocityExtrapolator {
 public:
  using Ptr = std::shared_ptr<VelocityExtrapolator>;
  using ConstPtr = std::shared_ptr<const VelocityExtrapolator>;
  using Variable = steam::traj::const_acc::Variable;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot,
                        const Eigen::Matrix<double, 6, 1>& ad) {
    return std::make_shared<VelocityExtrapolator>(time, knot, ad);
  }

  VelocityExtrapolator(const Time& time, const Variable::ConstPtr& knot,
                       const Eigen::Matrix<double, 6, 1>& ad)
      : steam::traj::const_acc::VelocityExtrapolator(time, knot) {
    const double tau = (time - knot->time()).seconds();
    Phi_ = getTran(tau, ad);
  }
};

}  // namespace singer
}  // namespace traj
}  // namespace steam