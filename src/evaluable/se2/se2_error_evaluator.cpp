#include "steam/evaluable/se2/se2_error_evaluator.hpp"

namespace steam {
namespace se2 {

auto SE2ErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_ab,
                                   const InType &T_ab_meas) -> Ptr {
  return std::make_shared<SE2ErrorEvaluator>(T_ab, T_ab_meas);
}

SE2ErrorEvaluator::SE2ErrorEvaluator(const Evaluable<InType>::ConstPtr &T_ab,
                                     const InType &T_ab_meas)
    : T_ab_(T_ab), T_ab_meas_(T_ab_meas) {}

bool SE2ErrorEvaluator::active() const { return T_ab_->active(); }

void SE2ErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_ab_->getRelatedVarKeys(keys);
}

auto SE2ErrorEvaluator::value() const -> OutType {
  return (T_ab_meas_ * T_ab_->value().inverse()).vec();
}

auto SE2ErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_ab_->forward();
  const auto value = (T_ab_meas_ * child->value().inverse()).vec();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void SE2ErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_ab_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    Eigen::MatrixXd new_lhs =
        lhs * (-1.0) * lgmath::se2::vec2jac(node->value());

    T_ab_->backward(new_lhs, child, jacs);
  }
}

SE2ErrorEvaluator::Ptr se2_error(
    const Evaluable<SE2ErrorEvaluator::InType>::ConstPtr &T_ab,
    const SE2ErrorEvaluator::InType &T_ab_meas) {
  return SE2ErrorEvaluator::MakeShared(T_ab, T_ab_meas);
}

}  // namespace se2
}  // namespace steam