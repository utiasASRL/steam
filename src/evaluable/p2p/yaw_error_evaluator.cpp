#include "steam/evaluable/p2p/yaw_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto YawErrorEvaluator::MakeShared(
    const double yaw_meas,
    const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
    const Evaluable<PoseInType>::ConstPtr &T_ms_curr) -> Ptr {
  return std::make_shared<YawErrorEvaluator>(yaw_meas, T_ms_prev, T_ms_curr);
}

YawErrorEvaluator::YawErrorEvaluator(
    const double yaw_meas,
    const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
    const Evaluable<PoseInType>::ConstPtr &T_ms_curr)
    : yaw_meas_(yaw_meas), T_ms_prev_(T_ms_prev), T_ms_curr_(T_ms_curr) {
    d_ = Eigen::Matrix<double, 1, 3>::Zero();
    d_(0, 2) = 1.0;
}

bool YawErrorEvaluator::active() const { return T_ms_prev_->active() || T_ms_curr_->active(); }

void YawErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_ms_prev_->getRelatedVarKeys(keys);
  T_ms_curr_->getRelatedVarKeys(keys);
}

auto YawErrorEvaluator::value() const -> OutType {
  // Form measured and predicted printegrated DCM: prev (p) curr (c)
  Eigen::Vector3d meas_vec(0.0, 0.0, yaw_meas_);
  const lgmath::so3::Rotation C_pc_meas(meas_vec);
  const lgmath::so3::Rotation C_cp_eval((T_ms_curr_->value().C_ba().inverse() * 
                                                T_ms_prev_->value().C_ba()).eval());
  // Return error
  return d_ * lgmath::so3::rot2vec((C_cp_eval * C_pc_meas).matrix());
}

auto YawErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = T_ms_prev_->forward();
  const auto child2 = T_ms_curr_->forward();

  const auto C_ms_prev = child1->value().C_ba();
  const auto C_ms_curr = child2->value().C_ba();

  // Form measured and predicted printegrated DCM: prev (p) curr (c)
  // clang-format off
  Eigen::Vector3d meas_vec(0.0, 0.0, yaw_meas_);
  const lgmath::so3::Rotation C_pc_meas(meas_vec);
  const lgmath::so3::Rotation C_cp_eval((C_ms_curr.transpose() * C_ms_prev).eval());
  OutType error = d_ * lgmath::so3::rot2vec((C_cp_eval * C_pc_meas).matrix());
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);
  return node;
}

void YawErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                       const Node<OutType>::Ptr &node,
                                       Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<PoseInType>>(node->at(1));

  if (T_ms_prev_->active()) {
    // Jacobian is just -1 for the yaw component
    Eigen::Matrix<double, 1, 6> jac = Eigen::Matrix<double, 1, 6>::Zero();
    jac(0, 5) = -1.0;
    T_ms_prev_->backward(lhs * jac, child1, jacs);
  }
  if (T_ms_curr_->active()) {
    // Jacobian is just 1 for the yaw component
    Eigen::Matrix<double, 1, 6> jac = Eigen::Matrix<double, 1, 6>::Zero();
    jac(0, 5) = 1.0;
    T_ms_curr_->backward(lhs * jac, child2, jacs);
  }
}

YawErrorEvaluator::Ptr yawError(
    const double yaw_meas,
    const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_prev,
    const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_curr) {
  return YawErrorEvaluator::MakeShared(yaw_meas, T_ms_prev, T_ms_curr);
}

}  // namespace p2p
}  // namespace steam