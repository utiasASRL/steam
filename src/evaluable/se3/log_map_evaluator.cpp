#include "steam/evaluable/se3/log_map_evaluator.hpp"

namespace steam {
namespace se3 {

auto LogMapEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &transform)
    -> Ptr {
  return std::make_shared<LogMapEvaluator>(transform);
}

LogMapEvaluator::LogMapEvaluator(const Evaluable<InType>::ConstPtr &transform)
    : transform_(transform) {}

bool LogMapEvaluator::active() const { return transform_->active(); }

auto LogMapEvaluator::value() const -> OutType {
  return transform_->value().vec();
}

auto LogMapEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = transform_->forward();
  const auto value = child->value().vec();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void LogMapEvaluator::backward(const Eigen::MatrixXd &lhs,
                               const Node<OutType>::Ptr &node,
                               Jacobians &jacs) const {
  if (transform_->active()) {
    Eigen::MatrixXd new_lhs = lhs * lgmath::se3::vec2jacinv(node->value());
    transform_->backward(
        new_lhs, std::static_pointer_cast<Node<InType>>(node->at(0)), jacs);
  }
}

LogMapEvaluator::Ptr tran2vec(
    const Evaluable<LogMapEvaluator::InType>::ConstPtr &transform) {
  return LogMapEvaluator::MakeShared(transform);
}

}  // namespace se3
}  // namespace steam