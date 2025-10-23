#include "steam/trajectory/const_vel_se2/evaluable/jinv_velocity_evaluator.hpp"

namespace steam {
namespace traj {
namespace const_vel_se2 {

auto JinvVelocityEvaluator::MakeShared(
    const Evaluable<XiInType>::ConstPtr& xi,
    const Evaluable<VelInType>::ConstPtr& velocity) -> Ptr {
  return std::make_shared<JinvVelocityEvaluator>(xi, velocity);
}

JinvVelocityEvaluator::JinvVelocityEvaluator(
    const Evaluable<XiInType>::ConstPtr& xi,
    const Evaluable<VelInType>::ConstPtr& velocity)
    : xi_(xi), velocity_(velocity) {}

bool JinvVelocityEvaluator::active() const {
  return xi_->active() || velocity_->active();
}

void JinvVelocityEvaluator::getRelatedVarKeys(KeySet& keys) const {
  xi_->getRelatedVarKeys(keys);
  velocity_->getRelatedVarKeys(keys);
}

auto JinvVelocityEvaluator::value() const -> OutType {
  return lgmath::se2::vec2jacinv(xi_->value()) * velocity_->value();
}

auto JinvVelocityEvaluator::forward() const -> Node<OutType>::Ptr {
  //
  const auto child1 = xi_->forward();
  const auto child2 = velocity_->forward();

  //
  const auto value = lgmath::se2::vec2jacinv(child1->value()) * child2->value();

  //
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void JinvVelocityEvaluator::backward(const Eigen::MatrixXd& lhs,
                                     const Node<OutType>::Ptr& node,
                                     Jacobians& jacs) const {
  const auto child1 = std::static_pointer_cast<Node<XiInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<VelInType>>(node->at(1));

  if (xi_->active()) {
    xi_->backward((0.5) * lhs * lgmath::se2::curlyhat(child2->value()), child1,
                  jacs);
  }

  if (velocity_->active()) {
    velocity_->backward(lhs * lgmath::se2::vec2jacinv(child1->value()), child2,
                        jacs);
  }
}

JinvVelocityEvaluator::Ptr jinv_velocity(
    const Evaluable<JinvVelocityEvaluator::XiInType>::ConstPtr& xi,
    const Evaluable<JinvVelocityEvaluator::VelInType>::ConstPtr& velocity) {
  return JinvVelocityEvaluator::MakeShared(xi, velocity);
}

}  // namespace const_vel_se2
}  // namespace traj
}  // namespace steam