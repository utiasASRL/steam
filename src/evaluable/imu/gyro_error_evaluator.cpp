#include "steam/evaluable/imu/gyro_error_evaluator.hpp"
#include <iostream>
#include <lgmath/so3/Operations.hpp>

namespace steam {
namespace imu {

auto GyroErrorEvaluator::MakeShared(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
    -> Ptr {
  return std::make_shared<GyroErrorEvaluator>(velocity, bias, gyro_meas);
}

GyroErrorEvaluator::GyroErrorEvaluator(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias, const ImuInType &gyro_meas)
    : velocity_(velocity), bias_(bias), gyro_meas_(gyro_meas) {
  const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
  Dw_ = I.block<3, 6>(3, 0);
}

bool GyroErrorEvaluator::active() const {
  return velocity_->active() || bias_->active();
}

void GyroErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  velocity_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
}

auto GyroErrorEvaluator::value() const -> OutType {
  // clang-format off
  OutType error = gyro_meas_ + Dw_ * velocity_->value() - bias_->value().block<3, 1>(3, 0);
  return error;
  // clang-format on
}

auto GyroErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = velocity_->forward();
  const auto child2 = bias_->forward();
  const auto w_mv_in_v = child1->value();
  const auto b = child2->value();

  // clang-format off
  OutType error = gyro_meas_ + Dw_ * w_mv_in_v - b.block<3, 1>(3, 0);
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void GyroErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                  const Node<OutType>::Ptr &node,
                                  Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<BiasInType>>(node->at(1));

  if (velocity_->active()) {
    Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
    jac.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    velocity_->backward(lhs * jac, child1, jacs);
  }
  if (bias_->active()) {
    Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
    jac.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity() * -1;
    bias_->backward(lhs * jac, child2, jacs);
  }
  // clang-format on
}

Eigen::Matrix<double, 3, 6> GyroErrorEvaluator::getJacobianVelocity() const {
  Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
  jac.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
  return jac;
}

Eigen::Matrix<double, 3, 6> GyroErrorEvaluator::getJacobianBias() const {
  Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
  jac.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity() * -1;
  return jac;
}

GyroErrorEvaluator::Ptr GyroError(
    const Evaluable<GyroErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<GyroErrorEvaluator::BiasInType>::ConstPtr &bias,
    const GyroErrorEvaluator::ImuInType &gyro_meas) {
  return GyroErrorEvaluator::MakeShared(velocity, bias, gyro_meas);
}

}  // namespace imu
}  // namespace steam