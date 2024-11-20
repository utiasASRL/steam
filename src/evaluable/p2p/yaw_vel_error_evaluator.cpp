#include "steam/evaluable/p2p/yaw_vel_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto YawVelErrorEvaluator::MakeShared(
    const Eigen::Matrix<double, 1, 1> vel_meas,
    const Evaluable<InType>::ConstPtr &w_iv_inv) -> Ptr {
  return std::make_shared<YawVelErrorEvaluator>(vel_meas, w_iv_inv);
}

YawVelErrorEvaluator::YawVelErrorEvaluator(
    const Eigen::Matrix<double, 1, 1> vel_meas,
    const Evaluable<InType>::ConstPtr &w_iv_inv)
    : vel_meas_(vel_meas), w_iv_inv_(w_iv_inv) {
    D_ << 0, 0, 0, 0, 0, 1;
}

bool YawVelErrorEvaluator::active() const { return w_iv_inv_->active();}

void YawVelErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  w_iv_inv_->getRelatedVarKeys(keys);
}

auto YawVelErrorEvaluator::value() const -> OutType {
  // Return error
  Eigen::Matrix<double,1,1> res;
  res << vel_meas_ - D_ * w_iv_inv_->value();
  return res; 
}

auto YawVelErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = w_iv_inv_->forward();
  // clang-format off
  OutType error = vel_meas_ - D_ * child1->value();
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  return node;
}

void YawVelErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                       const Node<OutType>::Ptr &node,
                                       Jacobians &jacs) const {
  if (w_iv_inv_->active()) {
    const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
    Eigen::Matrix<double, 1, 6> jac = -D_;
    w_iv_inv_->backward(lhs * jac, child1, jacs);
  }
}

YawVelErrorEvaluator::Ptr velError(
    const Eigen::Matrix<double, 1, 1> vel_meas,
    const Evaluable<YawVelErrorEvaluator::InType>::ConstPtr &w_iv_inv) {
  return YawVelErrorEvaluator::MakeShared(vel_meas, w_iv_inv);
}

}  // namespace p2p
}  // namespace steam