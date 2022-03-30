#include "steam/trajectory/const_vel/pose_interpolator.hpp"

namespace steam {
namespace traj {
namespace const_vel {

PoseInterpolator::Ptr PoseInterpolator::MakeShared(
    const Time& time, const Variable::ConstPtr& knot1,
    const Variable::ConstPtr& knot2) {
  return std::make_shared<PoseInterpolator>(time, knot1, knot2);
}

PoseInterpolator::PoseInterpolator(const Time& time,
                                   const Variable::ConstPtr& knot1,
                                   const Variable::ConstPtr& knot2)
    : knot1_(knot1), knot2_(knot2) {
  // Calculate time constants
  double tau = (time - knot1->getTime()).seconds();
  double T = (knot2->getTime() - knot1->getTime()).seconds();
  double ratio = tau / T;
  double ratio2 = ratio * ratio;
  double ratio3 = ratio2 * ratio;

  // Calculate 'psi' interpolation values
  psi11_ = 3.0 * ratio2 - 2.0 * ratio3;
  psi12_ = tau * (ratio2 - ratio);
  psi21_ = 6.0 * (ratio - ratio2) / T;
  psi22_ = 3.0 * ratio2 - 2.0 * ratio;

  // Calculate 'lambda' interpolation values
  lambda11_ = 1.0 - psi11_;
  lambda12_ = tau - T * psi11_ - psi12_;
  lambda21_ = -psi21_;
  lambda22_ = 1.0 - T * psi21_ - psi22_;
}

bool PoseInterpolator::active() const {
  return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
         knot2_->getPose()->active() || knot2_->getVelocity()->active();
}

auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
  //
  const auto pose1_child = knot1_->getPose()->forward();
  const auto vel1_child = knot1_->getVelocity()->forward();
  const auto pose2_child = knot2_->getPose()->forward();
  const auto vel2_child = knot2_->getVelocity()->forward();

  // Get relative matrix info
  const auto T_21 = pose2_child->value() / pose1_child->value();
  // Get se3 algebra of relative matrix
  const auto xi_21 = T_21.vec();
  // Calculate the 6x6 associated Jacobian
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  // Calculate interpolated relative se3 algebra
  Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * vel1_child->value() +
                                      psi11_ * xi_21 +
                                      psi12_ * J_21_inv * vel2_child->value();
  // Calculate interpolated relative transformation matrix
  lgmath::se3::Transformation T_i1(xi_i1);
  // Return `global' interpolated transform
  const auto T_ik = T_i1 * pose1_child->value();

  //
  const auto node = Node<OutType>::MakeShared(T_ik);
  node->addChild(pose1_child);
  node->addChild(vel1_child);
  node->addChild(pose2_child);
  node->addChild(vel2_child);

  return node;
}

void PoseInterpolator::backward(const Eigen::MatrixXd& lhs,
                                const Node<OutType>::Ptr& node,
                                Jacobians& jacs) const {
  if (!active()) return;

  // clang-format off
  const auto pose1_child = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
  const auto vel1_child = std::static_pointer_cast<Node<InVelType>>(node->at(1));
  const auto pose2_child = std::static_pointer_cast<Node<InPoseType>>(node->at(2));
  const auto vel2_child = std::static_pointer_cast<Node<InVelType>>(node->at(3));
  // clang-format on

  // Get relative matrix info
  const auto T_21 = pose2_child->value() / pose1_child->value();
  // Get se3 algebra of relative matrix
  const auto xi_21 = T_21.vec();
  // Calculate the 6x6 associated Jacobian
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  // Calculate interpolated relative se3 algebra
  Eigen::Matrix<double, 6, 1> xi_i1 = lambda12_ * vel1_child->value() +
                                      psi11_ * xi_21 +
                                      psi12_ * J_21_inv * vel2_child->value();
  // Calculate interpolated relative transformation matrix
  lgmath::se3::Transformation T_i1(xi_i1);
  // Calculate the 6x6 Jacobian associated with the interpolated relative
  // transformation matrix
  const auto J_i1 = lgmath::se3::vec2jac(xi_i1);

  // Knot 1 transform
  if (knot1_->getPose()->active() || knot2_->getPose()->active()) {
    // Precompute matrix
    Eigen::Matrix<double, 6, 6> w =
        psi11_ * J_i1 * J_21_inv +
        0.5 * psi12_ * J_i1 * lgmath::se3::curlyhat(vel2_child->value()) *
            J_21_inv;

    // Check if transform1 is active
    if (knot1_->getPose()->active()) {
      Eigen::Matrix<double, 6, 6> jacobian =
          (-1) * w * T_21.adjoint() + T_i1.adjoint();
      knot1_->getPose()->backward(lhs * jacobian, pose1_child, jacs);
    }

    // Check if transform2 is active
    if (knot2_->getPose()->active()) {
      knot2_->getPose()->backward(lhs * w, pose2_child, jacs);
    }
  }

  // 6 x 6 Velocity Jacobian 1
  if (knot1_->getVelocity()->active()) {
    // Add Jacobian
    Eigen::Matrix<double, 6, 6> jacobian = lambda12_ * J_i1;
    knot1_->getVelocity()->backward(lhs * jacobian, vel1_child, jacs);
  }

  // 6 x 6 Velocity Jacobian 2
  if (knot2_->getVelocity()->active()) {
    // Add Jacobian
    Eigen::Matrix<double, 6, 6> jacobian = psi12_ * J_i1 * J_21_inv;
    knot2_->getVelocity()->backward(lhs * jacobian, vel2_child, jacs);
  }
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam