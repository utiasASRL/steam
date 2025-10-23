#include "steam/evaluable/imu/gyro_error_evaluator_se2.hpp"
#include <lgmath/so2/Operations.hpp>

namespace steam {
namespace imu {

auto GyroErrorEvaluatorSE2::MakeShared(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
    -> Ptr {
  return std::make_shared<GyroErrorEvaluatorSE2>(velocity, bias, gyro_meas);
}

GyroErrorEvaluatorSE2::GyroErrorEvaluatorSE2(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
    : velocity_(velocity), bias_(bias), gyro_meas_(gyro_meas) {
  jac_vel_(0, 2) = 1;
  jac_bias_(0, 2) = -1;
}

bool GyroErrorEvaluatorSE2::active() const {
  return velocity_->active() || bias_->active();
}

void GyroErrorEvaluatorSE2::getRelatedVarKeys(KeySet &keys) const {
  velocity_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
}

auto GyroErrorEvaluatorSE2::value() const -> OutType {
  // clang-format off
  OutType error(gyro_meas_ + (velocity_->value())(2, 0) - bias_->value()(2, 0));
  return error;
  // clang-format on
}

auto GyroErrorEvaluatorSE2::forward() const -> Node<OutType>::Ptr {
  const auto child1 = velocity_->forward();
  const auto child2 = bias_->forward();
  const auto w_mv_in_v = child1->value();
  const auto b = child2->value();

  // clang-format off
  OutType error(gyro_meas_ + w_mv_in_v(2, 0) - b(2, 0));
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void GyroErrorEvaluatorSE2::backward(const Eigen::MatrixXd &lhs,
                                     const Node<OutType>::Ptr &node,
                                     Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<BiasInType>>(node->at(1));

  if (velocity_->active()) {
    velocity_->backward(lhs * jac_vel_, child1, jacs);
  }
  if (bias_->active()) {
    bias_->backward(lhs * jac_bias_, child2, jacs);
  }
  // clang-format on
}

GyroErrorEvaluatorSE2::Ptr GyroErrorSE2(
    const Evaluable<GyroErrorEvaluatorSE2::VelInType>::ConstPtr &velocity,
    const Evaluable<GyroErrorEvaluatorSE2::BiasInType>::ConstPtr &bias,
    const GyroErrorEvaluatorSE2::ImuInType &gyro_meas) {
  return GyroErrorEvaluatorSE2::MakeShared(velocity, bias, gyro_meas);
}

}  // namespace imu
}  // namespace steam