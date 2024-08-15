#include "steam/evaluable/p2p/vel_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto VelErrorEvaluator::MakeShared(
    const Eigen::Vector2d vel_meas,
    const Evaluable<InType>::ConstPtr &w_iv_inv) -> Ptr {
  return std::make_shared<VelErrorEvaluator>(vel_meas, w_iv_inv);
}

VelErrorEvaluator::VelErrorEvaluator(
    const Eigen::Vector2d vel_meas,
    const Evaluable<InType>::ConstPtr &w_iv_inv)
    : vel_meas_(vel_meas), w_iv_inv_(w_iv_inv) {
    D_ = Eigen::Matrix<double, 2, 6>::Zero();
    D_(0, 0) = 1.0; // Pick off forward velocity
    D_(1, 1) = 1.0; // Pick off lateral velocity
}

bool VelErrorEvaluator::active() const { return w_iv_inv_->active();}

void VelErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  w_iv_inv_->getRelatedVarKeys(keys);
}

auto VelErrorEvaluator::value() const -> OutType {
  // Return error
  return vel_meas_ - D_ * w_iv_inv_->value();
}

auto VelErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = w_iv_inv_->forward();
  // clang-format off
  OutType error = vel_meas_ - D_ * child1->value();
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  return node;
}

void VelErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                       const Node<OutType>::Ptr &node,
                                       Jacobians &jacs) const {
  if (w_iv_inv_->active()) {
    const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
    Eigen::Matrix<double, 2, 6> jac = -D_;
    w_iv_inv_->backward(lhs * jac, child1, jacs);
  }
}

VelErrorEvaluator::Ptr velError(
    const Eigen::Vector2d vel_meas,
    const Evaluable<VelErrorEvaluator::InType>::ConstPtr &w_iv_inv) {
  return VelErrorEvaluator::MakeShared(vel_meas, w_iv_inv);
}

}  // namespace p2p
}  // namespace steam