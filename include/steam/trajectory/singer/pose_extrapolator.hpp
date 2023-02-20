#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_acc/pose_extrapolator.hpp"
#include "steam/trajectory/const_acc/variable.hpp"
#include "steam/trajectory/singer/helper.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace singer {

class PoseExtrapolator : public steam::traj::const_acc::PoseExtrapolator {
 public:
  using Ptr = std::shared_ptr<PoseExtrapolator>;
  using ConstPtr = std::shared_ptr<const PoseExtrapolator>;
  using Variable = steam::traj::const_acc::Variable;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot,
                        const Eigen::Matrix<double, 6, 1>& ad) {
    return std::make_shared<PoseExtrapolator>(time, knot, ad);
  }

  PoseExtrapolator(const Time& time, const Variable::ConstPtr& knot,
                   const Eigen::Matrix<double, 6, 1>& ad)
      : steam::traj::const_acc::PoseExtrapolator(time, knot) {
    const double tau = (time - knot->time()).seconds();
    Phi_ = getTran(tau, ad);
  }
};

}  // namespace singer
}  // namespace traj
}  // namespace steam