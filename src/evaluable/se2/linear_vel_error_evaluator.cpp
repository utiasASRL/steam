#include "steam/evaluable/se2/linear_vel_error_evaluator.hpp"

namespace steam {
namespace se2 {

auto LinearVelErrorEvaluator::MakeShared(
    const Eigen::Vector2d& vel_meas,
    const Evaluable<InType>::ConstPtr &w_iv_inv) -> Ptr {
  return std::make_shared<LinearVelErrorEvaluator>(vel_meas, w_iv_inv);
}

LinearVelErrorEvaluator::LinearVelErrorEvaluator(
    const Eigen::Vector2d& vel_meas,
    const Evaluable<InType>::ConstPtr &w_iv_inv)
    : vel_meas_(vel_meas), w_iv_inv_(w_iv_inv) {
    D_.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();
}

bool LinearVelErrorEvaluator::active() const { return w_iv_inv_->active();}

void LinearVelErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  w_iv_inv_->getRelatedVarKeys(keys);
}

auto LinearVelErrorEvaluator::value() const -> OutType {
  return vel_meas_ - D_ * w_iv_inv_->value();
}

auto LinearVelErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = w_iv_inv_->forward();
  OutType error = vel_meas_ - D_ * child1->value();
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  return node;
}

void LinearVelErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                       const Node<OutType>::Ptr &node,
                                       Jacobians &jacs) const {
  if (w_iv_inv_->active()) {
    const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
    Eigen::Matrix<double, 2, 3> jac = -D_;
    w_iv_inv_->backward(lhs * jac, child1, jacs);
  }
}

LinearVelErrorEvaluator::Ptr linearVelError(
    const Eigen::Vector2d& vel_meas,
    const Evaluable<LinearVelErrorEvaluator::InType>::ConstPtr &w_iv_inv) {
  return LinearVelErrorEvaluator::MakeShared(vel_meas, w_iv_inv);
}

}  // namespace se2
}  // namespace steam