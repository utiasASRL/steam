#include "steam/trajectory/const_vel/pose_extrapolator.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto PoseExtrapolator::MakeShared(const Time& time,
                                  const Evaluable<InType>::ConstPtr& velocity)
    -> Ptr {
  return std::make_shared<PoseExtrapolator>(time, velocity);
}

PoseExtrapolator::PoseExtrapolator(const Time& time,
                                   const Evaluable<InType>::ConstPtr& velocity)
    : time_(time), velocity_(velocity) {}

bool PoseExtrapolator::active() const { return velocity_->active(); }

auto PoseExtrapolator::value() const -> OutType {
  return OutType(
      Eigen::Matrix<double, 6, 1>(time_.seconds() * velocity_->value()));
}

auto PoseExtrapolator::forward() const -> Node<OutType>::Ptr {
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

void PoseExtrapolator::backward(const Eigen::MatrixXd& lhs,
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

}  // namespace const_vel
}  // namespace traj
}  // namespace steam