#include "steam/trajectory/traj_pose_extrapolator.hpp"

namespace steam {
namespace traj {

auto TrajPoseExtrapolator::MakeShared(
    const Time& time, const Evaluable<InType>::ConstPtr& velocity) -> Ptr {
  return std::make_shared<TrajPoseExtrapolator>(time, velocity);
}

TrajPoseExtrapolator::TrajPoseExtrapolator(
    const Time& time, const Evaluable<InType>::ConstPtr& velocity)
    : time_(time), velocity_(velocity) {}

bool TrajPoseExtrapolator::active() const { return velocity_->active(); }

auto TrajPoseExtrapolator::forward() const -> Node<OutType>::Ptr {
  //
  const auto child = velocity_->forward();

  //
  Eigen::Matrix<double, 6, 1> xi = time_.seconds() * child->value();
  OutType T(xi);

  //
  const auto node = Node<OutType>::MakeShared(T);
  node->addChild(child);

  return node;
}

void TrajPoseExtrapolator::backward(const Eigen::MatrixXd& lhs,
                                    const Node<OutType>::Ptr& node,
                                    Jacobians& jacs) const {
  if (velocity_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    // Make jacobian
    Eigen::Matrix<double, 6, 1> xi = time_.seconds() * child->value();
    Eigen::Matrix<double, 6, 6> jac =
        time_.seconds() * lgmath::se3::vec2jac(xi);
    // backward
    velocity_->backward(lhs * jac, child, jacs);
  }
}

}  // namespace traj
}  // namespace steam
