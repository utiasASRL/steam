#include "steam/evaluable/se3/se3_error_evaluator.hpp"

namespace steam {
namespace se3 {

auto SE3ErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_ab,
                                   const InType &T_ab_meas) -> Ptr {
  return std::make_shared<SE3ErrorEvaluator>(T_ab, T_ab_meas);
}

SE3ErrorEvaluator::SE3ErrorEvaluator(const Evaluable<InType>::ConstPtr &T_ab,
                                     const InType &T_ab_meas)
    : T_ab_(T_ab), T_ab_meas_(T_ab_meas) {}

bool SE3ErrorEvaluator::active() const { return T_ab_->active(); }

void SE3ErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_ab_->getRelatedVarKeys(keys);
}

auto SE3ErrorEvaluator::value() const -> OutType {
  return (T_ab_meas_ * T_ab_->value().inverse()).vec();
}

auto SE3ErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_ab_->forward();
  const auto value = (T_ab_meas_ * child->value().inverse()).vec();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void SE3ErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_ab_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    // Eigen::MatrixXd new_lhs = lhs * lgmath::se3::vec2jacinv(node->value()) *
    //                           (-1.0) *
    //                           (T_ab_meas_ *
    //                           child->value().inverse()).adjoint();
    Eigen::MatrixXd new_lhs =
        lhs * (-1.0) * lgmath::se3::vec2jac(node->value());

    T_ab_->backward(new_lhs, child, jacs);
  }
}

SE3ErrorEvaluator::Ptr se3_error(
    const Evaluable<SE3ErrorEvaluator::InType>::ConstPtr &T_ab,
    const SE3ErrorEvaluator::InType &T_ab_meas) {
  return SE3ErrorEvaluator::MakeShared(T_ab, T_ab_meas);
}

}  // namespace se3
}  // namespace steam