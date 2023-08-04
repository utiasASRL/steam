#include "steam/evaluable/imu/imu_error_evaluator.hpp"
#include <lgmath/so3/Operations.hpp>

namespace steam {
namespace imu {

auto IMUErrorEvaluator::MakeShared(
    const Evaluable<PoseInType>::ConstPtr &transform,
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<AccInType>::ConstPtr &acceleration,
    const Evaluable<BiasInType>::ConstPtr &bias,
    const Evaluable<GravityInType>::ConstPtr &gravity,
    const ImuInType &imu_meas) -> Ptr {
  return std::make_shared<IMUErrorEvaluator>(transform, velocity, acceleration,
                                             bias, gravity, imu_meas);
}

IMUErrorEvaluator::IMUErrorEvaluator(
    const Evaluable<PoseInType>::ConstPtr &transform,
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<AccInType>::ConstPtr &acceleration,
    const Evaluable<BiasInType>::ConstPtr &bias,
    const Evaluable<GravityInType>::ConstPtr &gravity,
    const ImuInType &imu_meas)
    : transform_(transform),
      velocity_(velocity),
      acceleration_(acceleration),
      bias_(bias),
      gravity_(gravity),
      imu_meas_(imu_meas) {
  const Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
  Da_ = I.block<3, 6>(0, 0);
  Dw_ = I.block<3, 6>(3, 0);
  DT_ = Eigen::Matrix<double, 3, 3>::Identity().block<3, 2>(0, 0);
}

bool IMUErrorEvaluator::active() const {
  return transform_->active() || velocity_->active() ||
         acceleration_->active() || bias_->active() || gravity_->active();
}

void IMUErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  transform_->getRelatedVarKeys(keys);
  velocity_->getRelatedVarKeys(keys);
  acceleration_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
  gravity_->getRelatedVarKeys(keys);
}

auto IMUErrorEvaluator::value() const -> OutType {
  // clang-format off
  const Eigen::Matrix3d C_vi = transform_->value().C_ba();
  OutType error = Eigen::Matrix<double, 6, 1>::Zero();
  error.block<3, 1>(0, 0) = imu_meas_.block<3, 1>(0, 0) + Da_ * acceleration_->value() + C_vi * gravity_->value() - bias_->value().block<3, 1>(0, 0);
  error.block<3, 1>(3, 0) = imu_meas_.block<3, 1>(3, 0) + Dw_ * velocity_->value() - bias_->value().block<3, 1>(3, 0);
  return error;
  // clang-format on
}

auto IMUErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = transform_->forward();
  const auto child2 = velocity_->forward();
  const auto child3 = acceleration_->forward();
  const auto child4 = bias_->forward();
  const auto child5 = gravity_->forward();

  const auto C_vi = child1->value().C_ba();
  const auto w_iv_in_v = child2->value();
  const auto w_iv_in_v_dot = child3->value();
  const auto b = child4->value();
  const auto g = child5->value();

  // clang-format off
  OutType error = Eigen::Matrix<double, 6, 1>::Zero();
  error.block<3, 1>(0, 0) = imu_meas_.block<3, 1>(0, 0) + Da_ * w_iv_in_v_dot + C_vi * g - b.block<3, 1>(0, 0);
  error.block<3, 1>(3, 0) = imu_meas_.block<3, 1>(3, 0) + Dw_ * w_iv_in_v - b.block<3, 1>(3, 0);
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);
  node->addChild(child3);
  node->addChild(child4);
  node->addChild(child5);

  return node;
}

void IMUErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<VelInType>>(node->at(1));
  const auto child3 = std::static_pointer_cast<Node<AccInType>>(node->at(2));
  const auto child4 = std::static_pointer_cast<Node<BiasInType>>(node->at(3));
  const auto child5 =
      std::static_pointer_cast<Node<GravityInType>>(node->at(4));

  // clang-format off
  if (transform_->active()) {
    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
    jac.block<3, 3>(0, 3) = -1 * lgmath::so3::hat(child1->value().C_ba() * child5->value());
    transform_->backward(lhs * jac, child1, jacs);
  }

  if (velocity_->active()) {
    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
    jac.block<3, 3>(3, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    velocity_->backward(lhs * jac, child2, jacs);
  }

  if (acceleration_->active()) {
    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
    jac.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();
    acceleration_->backward(lhs * jac, child3, jacs);
  }

  if (bias_->active()) {
    const Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Identity() * -1;
    bias_->backward(lhs * jac, child4, jacs);
  }

  if (gravity_->active()) {
     Eigen::Matrix<double, 6, 2> jac = Eigen::Matrix<double, 6, 2>::Zero();
     jac.block<3, 2>(0, 0) = -1 * child1->value().C_ba() * lgmath::so3::hat(child5->value()) * DT_;
     gravity_->backward(lhs * jac, child5, jacs);
  }
  // clang-format on
}

IMUErrorEvaluator::Ptr imuError(
    const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform,
    const Evaluable<IMUErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<IMUErrorEvaluator::AccInType>::ConstPtr &acceleration,
    const Evaluable<IMUErrorEvaluator::BiasInType>::ConstPtr &bias,
    const Evaluable<IMUErrorEvaluator::GravityInType>::ConstPtr &gravity,
    const IMUErrorEvaluator::ImuInType &imu_meas) {
  return IMUErrorEvaluator::MakeShared(transform, velocity, acceleration, bias,
                                       gravity, imu_meas);
}

}  // namespace imu
}  // namespace steam