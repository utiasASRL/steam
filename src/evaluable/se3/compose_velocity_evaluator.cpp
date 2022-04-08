#include "steam/evaluable/se3/compose_velocity_evaluator.hpp"

namespace steam {
namespace se3 {

auto ComposeVelocityEvaluator::MakeShared(
    const Evaluable<PoseInType>::ConstPtr &transform,
    const Evaluable<VelInType>::ConstPtr &velocity) -> Ptr {
  return std::make_shared<ComposeVelocityEvaluator>(transform, velocity);
}

ComposeVelocityEvaluator::ComposeVelocityEvaluator(
    const Evaluable<PoseInType>::ConstPtr &transform,
    const Evaluable<VelInType>::ConstPtr &velocity)
    : transform_(transform), velocity_(velocity) {}

bool ComposeVelocityEvaluator::active() const {
  return transform_->active() || velocity_->active();
}

auto ComposeVelocityEvaluator::value() const -> OutType {
  return lgmath::se3::tranAd(transform_->value().matrix()) * velocity_->value();
}

auto ComposeVelocityEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = transform_->forward();
  const auto child2 = velocity_->forward();
  const auto value =
      lgmath::se3::tranAd(child1->value().matrix()) * child2->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);
  return node;
}

void ComposeVelocityEvaluator::backward(const Eigen::MatrixXd &lhs,
                                        const Node<OutType>::Ptr &node,
                                        Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<VelInType>>(node->at(1));

  if (transform_->active()) {
    transform_->backward((-1) * lhs * lgmath::se3::curlyhat(node->value()),
                         child1, jacs);
  }

  if (velocity_->active()) {
    velocity_->backward(lhs * lgmath::se3::tranAd(child1->value().matrix()),
                        child2, jacs);
  }
}

ComposeVelocityEvaluator::Ptr compose_velocity(
    const Evaluable<ComposeVelocityEvaluator::PoseInType>::ConstPtr &transform,
    const Evaluable<ComposeVelocityEvaluator::VelInType>::ConstPtr &velocity) {
  return ComposeVelocityEvaluator::MakeShared(transform, velocity);
}

}  // namespace se3
}  // namespace steam