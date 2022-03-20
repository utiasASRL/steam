#include "steam/evaluable/se3/compose_evaluator.hpp"

namespace steam {
namespace se3 {

auto ComposeEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &transform1,
                                  const Evaluable<InType>::ConstPtr &transform2)
    -> Ptr {
  return std::make_shared<ComposeEvaluator>(transform1, transform2);
}

ComposeEvaluator::ComposeEvaluator(
    const Evaluable<InType>::ConstPtr &transform1,
    const Evaluable<InType>::ConstPtr &transform2)
    : transform1_(transform1), transform2_(transform2) {}

bool ComposeEvaluator::active() const {
  return transform1_->active() || transform2_->active();
}

auto ComposeEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = transform1_->forward();
  const auto child2 = transform2_->forward();
  const auto value = child1->value() * child2->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);
  return node;
}

void ComposeEvaluator::backward(const Eigen::MatrixXd &lhs,
                                const Node<OutType>::Ptr &node,
                                Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<InType>>(node->at(1));

  if (transform1_->active()) {
    transform1_->backward(lhs, child1, jacs);
  }

  if (transform2_->active()) {
    Eigen::MatrixXd new_lhs = lhs * child1->value().adjoint();
    transform2_->backward(new_lhs, child2, jacs);
  }
}

ComposeEvaluator::Ptr compose(
    const Evaluable<ComposeEvaluator::InType>::ConstPtr &transform1,
    const Evaluable<ComposeEvaluator::InType>::ConstPtr &transform2) {
  return ComposeEvaluator::MakeShared(transform1, transform2);
}

}  // namespace se3
}  // namespace steam