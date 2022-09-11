#include "steam/evaluable/se3/exp_map_evaluator.hpp"

namespace steam {
namespace se3 {

auto ExpMapEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &xi) -> Ptr {
  return std::make_shared<ExpMapEvaluator>(xi);
}

ExpMapEvaluator::ExpMapEvaluator(const Evaluable<InType>::ConstPtr &xi)
    : xi_(xi) {}

bool ExpMapEvaluator::active() const { return xi_->active(); }

void ExpMapEvaluator::getRelatedVarKeys(KeySet &keys) const {
  xi_->getRelatedVarKeys(keys);
}

auto ExpMapEvaluator::value() const -> OutType { return OutType(xi_->value()); }

auto ExpMapEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = xi_->forward();
  const auto value = OutType(child->value());
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void ExpMapEvaluator::backward(const Eigen::MatrixXd &lhs,
                               const Node<OutType>::Ptr &node,
                               Jacobians &jacs) const {
  if (xi_->active()) {
    Eigen::MatrixXd new_lhs = lhs * lgmath::se3::vec2jac(node->value().vec());
    xi_->backward(new_lhs, std::static_pointer_cast<Node<InType>>(node->at(0)),
                  jacs);
  }
}

ExpMapEvaluator::Ptr vec2tran(
    const Evaluable<ExpMapEvaluator::InType>::ConstPtr &xi) {
  return ExpMapEvaluator::MakeShared(xi);
}

}  // namespace se3
}  // namespace steam