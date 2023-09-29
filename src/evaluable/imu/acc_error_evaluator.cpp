#include "steam/evaluable/imu/acc_error_evaluator.hpp"
#include <iostream>
#include <lgmath/so3/Operations.hpp>

namespace steam {
namespace imu {

auto AccelerationErrorEvaluator::MakeShared(
    const Evaluable<PoseInType>::ConstPtr &transform,
    const Evaluable<AccInType>::ConstPtr &acceleration,
    const Evaluable<BiasInType>::ConstPtr &bias,
    const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
    const ImuInType &acc_meas) -> Ptr {
  return std::make_shared<AccelerationErrorEvaluator>(
      transform, acceleration, bias, transform_i_to_m, acc_meas);
}

AccelerationErrorEvaluator::AccelerationErrorEvaluator(
    const Evaluable<PoseInType>::ConstPtr &transform,
    const Evaluable<AccInType>::ConstPtr &acceleration,
    const Evaluable<BiasInType>::ConstPtr &bias,
    const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
    const ImuInType &acc_meas)
    : transform_(transform),
      acceleration_(acceleration),
      bias_(bias),
      transform_i_to_m_(transform_i_to_m),
      acc_meas_(acc_meas) {
  const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
  Da_ = I.block<3, 6>(0, 0);
  gravity_(2, 0) = -9.8042;
}

bool AccelerationErrorEvaluator::active() const {
  return transform_->active() || acceleration_->active() || bias_->active() ||
         transform_i_to_m_->active();
}

void AccelerationErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  transform_->getRelatedVarKeys(keys);
  acceleration_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
  transform_i_to_m_->getRelatedVarKeys(keys);
}

auto AccelerationErrorEvaluator::value() const -> OutType {
  // clang-format off
  const Eigen::Matrix3d C_vm = transform_->value().C_ba();
  const Eigen::Matrix3d C_mi = transform_i_to_m_->value().C_ba();
  OutType error = acc_meas_ + Da_ * acceleration_->value() + C_vm * C_mi * gravity_ - bias_->value().block<3, 1>(0, 0);
  // OutType error = acc_meas_ + Da_ * acceleration_->value() - bias_->value().block<3, 1>(0, 0);
  return error;
  // clang-format on
}

auto AccelerationErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = transform_->forward();
  const auto child2 = acceleration_->forward();
  const auto child3 = bias_->forward();
  const auto child4 = transform_i_to_m_->forward();

  const auto C_vm = child1->value().C_ba();
  const auto dw_mv_in_v = child2->value();
  const auto b = child3->value();
  const auto C_mi = child4->value().C_ba();

  // clang-format off
  OutType error = acc_meas_.block<3, 1>(0, 0) + Da_ * dw_mv_in_v + C_vm * C_mi * gravity_ - b.block<3, 1>(0, 0);
  // OutType error = acc_meas_ + Da_ * dw_mv_in_v - b.block<3, 1>(0, 0);
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);
  node->addChild(child3);
  node->addChild(child4);

  return node;
}

void AccelerationErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                          const Node<OutType>::Ptr &node,
                                          Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<AccInType>>(node->at(1));
  const auto child3 = std::static_pointer_cast<Node<BiasInType>>(node->at(2));
  const auto child4 = std::static_pointer_cast<Node<PoseInType>>(node->at(3));

  // clang-format off
  if (transform_->active()) {
    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
    jac.block<3, 3>(0, 3) = -1 * lgmath::so3::hat(child1->value().C_ba() * child4->value().C_ba() * gravity_);
    transform_->backward(lhs * jac, child1, jacs);
  }

  if (acceleration_->active()) {
    Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
    jac.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();
    acceleration_->backward(lhs * jac, child2, jacs);
  }

  if (bias_->active()) {
    Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
    jac.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity() * -1;
    bias_->backward(lhs * jac, child3, jacs);
  }

  if (transform_i_to_m_->active()) {
     Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
     jac.block<3, 3>(0, 3) = -1 * child1->value().C_ba() * lgmath::so3::hat(child4->value().C_ba() * gravity_);
     transform_i_to_m_->backward(lhs * jac, child4, jacs);
  }
  // clang-format on
}

AccelerationErrorEvaluator::Ptr AccelerationError(
    const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr
        &transform,
    const Evaluable<AccelerationErrorEvaluator::AccInType>::ConstPtr
        &acceleration,
    const Evaluable<AccelerationErrorEvaluator::BiasInType>::ConstPtr &bias,
    const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr
        &transform_i_to_m,
    const AccelerationErrorEvaluator::ImuInType &acc_meas) {
  return AccelerationErrorEvaluator::MakeShared(transform, acceleration, bias,
                                                transform_i_to_m, acc_meas);
}

}  // namespace imu
}  // namespace steam