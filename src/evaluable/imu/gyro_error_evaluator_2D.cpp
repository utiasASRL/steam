#include "steam/evaluable/imu/gyro_error_evaluator_2d.hpp"
#include <iostream>
#include <lgmath/so2/Operations.hpp>

namespace steam {
namespace imu {

auto GyroErrorEvaluator2D::MakeShared(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
    -> Ptr {
  return std::make_shared<GyroErrorEvaluator2D>(velocity, bias, gyro_meas);
}

GyroErrorEvaluator2D::GyroErrorEvaluator2D(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
    : velocity_(velocity), bias_(bias), gyro_meas_(gyro_meas) {
  jac_vel_(0, 2) = 1;
  jac_bias_(0, 2) = -1;
}

bool GyroErrorEvaluator2D::active() const {
  return velocity_->active() || bias_->active();
}

void GyroErrorEvaluator2D::getRelatedVarKeys(KeySet &keys) const {
  velocity_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
}

auto GyroErrorEvaluator2D::value() const -> OutType {
  // clang-format off
  OutType error(gyro_meas_ + (velocity_->value())(2, 0) - bias_->value()(2, 0));
  return error;
  // clang-format on
}

auto GyroErrorEvaluator2D::forward() const -> Node<OutType>::Ptr {
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

void GyroErrorEvaluator2D::backward(const Eigen::MatrixXd &lhs,
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

GyroErrorEvaluator2D::Ptr GyroError2D(
    const Evaluable<GyroErrorEvaluator2D::VelInType>::ConstPtr &velocity,
    const Evaluable<GyroErrorEvaluator2D::BiasInType>::ConstPtr &bias,
    const GyroErrorEvaluator2D::ImuInType &gyro_meas) {
  return GyroErrorEvaluator2D::MakeShared(velocity, bias, gyro_meas);
}

}  // namespace imu
}  // namespace steam