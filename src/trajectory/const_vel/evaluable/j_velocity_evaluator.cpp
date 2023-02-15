#include "steam/trajectory/const_vel/evaluable/j_velocity_evaluator.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto JVelocityEvaluator::MakeShared(
    const Evaluable<XiInType>::ConstPtr& xi,
    const Evaluable<VelInType>::ConstPtr& velocity) -> Ptr {
  return std::make_shared<JVelocityEvaluator>(xi, velocity);
}

JVelocityEvaluator::JVelocityEvaluator(
    const Evaluable<XiInType>::ConstPtr& xi,
    const Evaluable<VelInType>::ConstPtr& velocity)
    : xi_(xi), velocity_(velocity) {}

bool JVelocityEvaluator::active() const {
  return xi_->active() || velocity_->active();
}

void JVelocityEvaluator::getRelatedVarKeys(KeySet& keys) const {
  xi_->getRelatedVarKeys(keys);
  velocity_->getRelatedVarKeys(keys);
}

auto JVelocityEvaluator::value() const -> OutType {
  return lgmath::se3::vec2jac(xi_->value()) * velocity_->value();
}

auto JVelocityEvaluator::forward() const -> Node<OutType>::Ptr {
  //
  const auto child1 = xi_->forward();
  const auto child2 = velocity_->forward();

  //
  const auto value = lgmath::se3::vec2jac(child1->value()) * child2->value();

  //
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void JVelocityEvaluator::backward(const Eigen::MatrixXd& lhs,
                                  const Node<OutType>::Ptr& node,
                                  Jacobians& jacs) const {
  const auto child1 = std::static_pointer_cast<Node<XiInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<VelInType>>(node->at(1));

  if (xi_->active()) {
    xi_->backward((-0.5) * lhs * lgmath::se3::curlyhat(child2->value()), child1,
                  jacs);
  }

  if (velocity_->active()) {
    velocity_->backward(lhs * lgmath::se3::vec2jac(child1->value()), child2,
                        jacs);
  }
}

JVelocityEvaluator::Ptr j_velocity(
    const Evaluable<JVelocityEvaluator::XiInType>::ConstPtr& xi,
    const Evaluable<JVelocityEvaluator::VelInType>::ConstPtr& velocity) {
  return JVelocityEvaluator::MakeShared(xi, velocity);
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam