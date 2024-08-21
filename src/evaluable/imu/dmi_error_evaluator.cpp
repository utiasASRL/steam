#include "steam/evaluable/imu/dmi_error_evaluator.hpp"
#include <iostream>
#include <lgmath/so3/Operations.hpp>

namespace steam {
namespace imu {

auto DMIErrorEvaluator::MakeShared(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<ScaleInType>::ConstPtr &scale, const DMIInType &dmi_meas)
    -> Ptr {
  return std::make_shared<DMIErrorEvaluator>(velocity, scale, dmi_meas);
}

DMIErrorEvaluator::DMIErrorEvaluator(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<ScaleInType>::ConstPtr &scale, const DMIInType &dmi_meas)
    : velocity_(velocity), scale_(scale), dmi_meas_(dmi_meas) {
  jac_vel_(0, 0) = 1;
  jac_scale_(0, 0) = dmi_meas_;
}

bool DMIErrorEvaluator::active() const {
  return velocity_->active() || scale_->active();
}

void DMIErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  velocity_->getRelatedVarKeys(keys);
  scale_->getRelatedVarKeys(keys);
}

auto DMIErrorEvaluator::value() const -> OutType {
  // clang-format off
  OutType error(dmi_meas_ * (scale_->value())(0, 0) + (velocity_->value())(0, 0));
  return error;
  // clang-format on
}

auto DMIErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = velocity_->forward();
  const auto child2 = scale_->forward();
  const auto w_mv_in_v = child1->value();
  const auto scale = child2->value();

  // clang-format off
  OutType error(dmi_meas_ * scale(0, 0) + w_mv_in_v(0, 0));
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void DMIErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<ScaleInType>>(node->at(1));

  if (velocity_->active()) {
    velocity_->backward(lhs * jac_vel_, child1, jacs);
  }
  // if (scale_->active() && fabs(dmi_meas_) > 0.1) {
  //   scale_->backward(lhs * jac_scale_, child2, jacs);
  // }
  // clang-format on
}

DMIErrorEvaluator::Ptr DMIError(
    const Evaluable<DMIErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<DMIErrorEvaluator::ScaleInType>::ConstPtr &scale,
    const DMIErrorEvaluator::DMIInType &dmi_meas) {
  return DMIErrorEvaluator::MakeShared(velocity, scale, dmi_meas);
}

}  // namespace imu
}  // namespace steam
