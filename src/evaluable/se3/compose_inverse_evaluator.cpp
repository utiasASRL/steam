#include "steam/evaluable/se3/compose_inverse_evaluator.hpp"

namespace steam {
namespace se3 {

auto ComposeInverseEvaluator::MakeShared(
    const Evaluable<InType>::ConstPtr &transform1,
    const Evaluable<InType>::ConstPtr &transform2) -> Ptr {
  return std::make_shared<ComposeInverseEvaluator>(transform1, transform2);
}

ComposeInverseEvaluator::ComposeInverseEvaluator(
    const Evaluable<InType>::ConstPtr &transform1,
    const Evaluable<InType>::ConstPtr &transform2)
    : transform1_(transform1), transform2_(transform2) {}

bool ComposeInverseEvaluator::active() const {
  return transform1_->active() || transform2_->active();
}

void ComposeInverseEvaluator::getRelatedVarKeys(KeySet &keys) const {
  transform1_->getRelatedVarKeys(keys);
  transform2_->getRelatedVarKeys(keys);
}

auto ComposeInverseEvaluator::value() const -> OutType {
  return transform1_->value() * transform2_->value().inverse();
}

auto ComposeInverseEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = transform1_->forward();
  const auto child2 = transform2_->forward();
  const auto value = child1->value() * child2->value().inverse();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);
  return node;
}

void ComposeInverseEvaluator::backward(const Eigen::MatrixXd &lhs,
                                       const Node<OutType>::Ptr &node,
                                       Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<InType>>(node->at(1));

  if (transform1_->active()) {
    transform1_->backward(lhs, child1, jacs);
  }

  if (transform2_->active()) {
    const auto T_ba = child1->value() * child2->value().inverse();
    transform2_->backward(-lhs * T_ba.adjoint(), child2, jacs);
  }
}

ComposeInverseEvaluator::Ptr compose_rinv(
    const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr &transform1,
    const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr &transform2) {
  return ComposeInverseEvaluator::MakeShared(transform1, transform2);
}

}  // namespace se3
}  // namespace steam