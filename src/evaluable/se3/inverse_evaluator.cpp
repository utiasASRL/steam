#include "steam/evaluable/se3/inverse_evaluator.hpp"

namespace steam {
namespace se3 {

auto InverseEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &transform)
    -> Ptr {
  return std::make_shared<InverseEvaluator>(transform);
}

InverseEvaluator::InverseEvaluator(const Evaluable<InType>::ConstPtr &transform)
    : transform_(transform) {}

bool InverseEvaluator::active() const { return transform_->active(); }

auto InverseEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = transform_->forward();
  const auto value = child->value().inverse();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void InverseEvaluator::backward(const Eigen::MatrixXd &lhs,
                                const Node<OutType>::Ptr &node,
                                Jacobians &jacs) const {
  if (transform_->active()) {
    Eigen::MatrixXd new_lhs = (-1) * lhs * node->value().adjoint();
    transform_->backward(
        new_lhs, std::static_pointer_cast<Node<InType>>(node->at(0)), jacs);
  }
}

InverseEvaluator::Ptr inverse(
    const Evaluable<InverseEvaluator::InType>::ConstPtr &transform) {
  return InverseEvaluator::MakeShared(transform);
}

}  // namespace se3
}  // namespace steam