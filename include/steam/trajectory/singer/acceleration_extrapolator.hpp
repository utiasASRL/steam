#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_acc/acceleration_extrapolator.hpp"
#include "steam/trajectory/const_acc/variable.hpp"
#include "steam/trajectory/singer/helper.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace singer {

class AccelerationExtrapolator
    : public steam::traj::const_acc::AccelerationExtrapolator {
 public:
  using Ptr = std::shared_ptr<AccelerationExtrapolator>;
  using ConstPtr = std::shared_ptr<const AccelerationExtrapolator>;
  using Variable = steam::traj::const_acc::Variable;

  static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot,
                        const Eigen::Matrix<double, 6, 1>& ad) {
    return std::make_shared<AccelerationExtrapolator>(time, knot, ad);
  }

  AccelerationExtrapolator(const Time time, const Variable::ConstPtr& knot,
                           const Eigen::Matrix<double, 6, 1>& ad)
      : steam::traj::const_acc::AccelerationExtrapolator(time, knot) {
    const double tau = (time - knot->time()).seconds();
    Phi_ = getTran(tau, ad);
  }
};

}  // namespace singer
}  // namespace traj
}  // namespace steam
