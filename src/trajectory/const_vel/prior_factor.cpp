#include "steam/trajectory/const_vel/prior_factor.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto PriorFactor::MakeShared(const Variable::ConstPtr& knot1,
                             const Variable::ConstPtr& knot2) -> Ptr {
  return std::make_shared<PriorFactor>(knot1, knot2);
}

PriorFactor::PriorFactor(const Variable::ConstPtr& knot1,
                         const Variable::ConstPtr& knot2)
    : knot1_(knot1), knot2_(knot2) {}

bool PriorFactor::active() const {
  return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
         knot2_->getPose()->active() || knot2_->getVelocity()->active();
}

auto PriorFactor::forward() const -> Node<OutType>::Ptr {
  //
  const auto pose1_child = knot1_->getPose()->forward();
  const auto vel1_child = knot1_->getVelocity()->forward();
  const auto pose2_child = knot2_->getPose()->forward();
  const auto vel2_child = knot2_->getVelocity()->forward();

  // Precompute values
  const auto T_21 = pose2_child->value() / pose1_child->value();
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Compute error
  double deltaTime = (knot2_->getTime() - knot1_->getTime()).seconds();
  OutType error;
  error.head<6>() = xi_21 - deltaTime * vel1_child->value();
  error.tail<6>() = J_21_inv * vel2_child->value() - vel1_child->value();

  //
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(pose1_child);
  node->addChild(vel1_child);
  node->addChild(pose2_child);
  node->addChild(vel2_child);

  return node;
}

void PriorFactor::backward(const Eigen::MatrixXd& lhs,
                           const Node<OutType>::Ptr& node,
                           Jacobians& jacs) const {
  if (!active()) return;

  // clang-format off
  const auto pose1_child = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
  const auto vel1_child = std::static_pointer_cast<Node<InVelType>>(node->at(1));
  const auto pose2_child = std::static_pointer_cast<Node<InPoseType>>(node->at(2));
  const auto vel2_child = std::static_pointer_cast<Node<InVelType>>(node->at(3));
  // clang-format on

  // Compute intermediate values
  const auto T_21 = pose2_child->value() / pose1_child->value();
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  double deltaTime = (knot2_->getTime() - knot1_->getTime()).seconds();

  // Knot 1 transform
  if (knot1_->getPose()->active()) {
    const auto Jinv_12 = J_21_inv * T_21.adjoint();
    // Construct jacobian
    Eigen::Matrix<double, 12, 6> jacobian;
    jacobian.topRows<6>() = -Jinv_12;
    jacobian.bottomRows<6>() =
        -0.5 * lgmath::se3::curlyhat(vel2_child->value()) * Jinv_12;
    // Get Jacobians
    knot1_->getPose()->backward(lhs * jacobian, pose1_child, jacs);
  }

  // Knot 2 transform
  if (knot2_->getPose()->active()) {
    // Construct jacobian
    Eigen::Matrix<double, 12, 6> jacobian;
    jacobian.topRows<6>() = J_21_inv;
    jacobian.bottomRows<6>() =
        0.5 * lgmath::se3::curlyhat(vel2_child->value()) * J_21_inv;
    // Get Jacobians
    knot2_->getPose()->backward(lhs * jacobian, pose2_child, jacs);
  }

  // Knot 1 velocity
  if (knot1_->getVelocity()->active()) {
    // Construct Jacobian Object
    Eigen::Matrix<double, 12, 6> jacobian;
    jacobian.topRows<6>() =
        -deltaTime * Eigen::Matrix<double, 6, 6>::Identity();
    jacobian.bottomRows<6>() = -Eigen::Matrix<double, 6, 6>::Identity();
    // Get Jacobians
    knot1_->getVelocity()->backward(lhs * jacobian, vel1_child, jacs);
  }

  // Knot 2 velocity
  if (knot2_->getVelocity()->active()) {
    // Construct Jacobian Object
    Eigen::Matrix<double, 12, 6> jacobian;
    jacobian.topRows<6>() = Eigen::Matrix<double, 6, 6>::Zero();
    jacobian.bottomRows<6>() = J_21_inv;
    // Get Jacobians
    knot2_->getVelocity()->backward(lhs * jacobian, vel2_child, jacs);
  }
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam