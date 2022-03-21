#include "steam/trajectory/traj_velocity_interpolator.hpp"

namespace steam {
namespace traj {

TrajVelocityInterpolator::Ptr TrajVelocityInterpolator::MakeShared(
    const Time& time, const TrajVar::ConstPtr& knot1,
    const TrajVar::ConstPtr& knot2) {
  return std::make_shared<TrajVelocityInterpolator>(time, knot1, knot2);
}

TrajVelocityInterpolator::TrajVelocityInterpolator(
    const Time& time, const TrajVar::ConstPtr& knot1,
    const TrajVar::ConstPtr& knot2)
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

bool TrajVelocityInterpolator::active() const {
  return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
         knot2_->getPose()->active() || knot2_->getVelocity()->active();
}

auto TrajVelocityInterpolator::forward() const -> Node<OutType>::Ptr {
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
  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double, 6, 6> J_t1 = lgmath::se3::vec2jac(xi_i1);
  // Calculate interpolated relative se3 algebra
  Eigen::VectorXd xi_it =
      J_t1 * (lambda22_ * vel1_child->value() + psi21_ * xi_21 +
              psi22_ * J_21_inv * vel2_child->value());

  //
  const auto node = Node<OutType>::MakeShared(xi_it);
  node->addChild(pose1_child);
  node->addChild(vel1_child);
  node->addChild(pose2_child);
  node->addChild(vel2_child);

  return node;
}

void TrajVelocityInterpolator::backward(const Eigen::MatrixXd& lhs,
                                        const Node<OutType>::Ptr& node,
                                        Jacobians& jacs) const {
  throw std::runtime_error("Not implemented");
}

}  // namespace traj
}  // namespace steam
