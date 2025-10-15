#include "steam/evaluable/se2/radial_vel_error_evaluator.hpp"
#include <iostream>
#include <lgmath/so3/Operations.hpp>

namespace steam {
namespace se2 {

auto RadialVelErrorEvaluator::MakeShared(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias,
    const AzimuthInType &azimuth,
    const DopplerInType &doppler_meas)
    -> Ptr {
  return std::make_shared<RadialVelErrorEvaluator>(velocity, bias, azimuth, doppler_meas);
}

RadialVelErrorEvaluator::RadialVelErrorEvaluator(
    const Evaluable<VelInType>::ConstPtr &velocity,
    const Evaluable<BiasInType>::ConstPtr &bias,
    const AzimuthInType &azimuth,
    const DopplerInType &doppler_meas)
    : velocity_(velocity), bias_(bias), doppler_meas_(doppler_meas) {
  D_(0, 0) = cos(azimuth);
  D_(0, 1) = sin(azimuth);
}

bool RadialVelErrorEvaluator::active() const {
  return velocity_->active() || bias_->active();
}

void RadialVelErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  velocity_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
}

auto RadialVelErrorEvaluator::value() const -> OutType {
  // clang-format off
  OutType error(doppler_meas_ + D_ * (velocity_->value() + bias_->value()));
  return error;
  // clang-format on
}

auto RadialVelErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = velocity_->forward();
  const auto child2 = bias_->forward();
  const auto w_mv_in_v = child1->value();
  const auto b = child2->value();

  // clang-format off
  OutType error(doppler_meas_ + D_ * (velocity_->value() + bias_->value()));
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void RadialVelErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                     const Node<OutType>::Ptr &node,
                                     Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<VelInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<BiasInType>>(node->at(1));

  if (velocity_->active()) {
    velocity_->backward(lhs * D_, child1, jacs);
  }
  if (bias_->active()) {
    bias_->backward(lhs * D_, child2, jacs);
  }
  // clang-format on
}

RadialVelErrorEvaluator::Ptr radialVelError(
    const Evaluable<RadialVelErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<RadialVelErrorEvaluator::BiasInType>::ConstPtr &bias,
    const RadialVelErrorEvaluator::AzimuthInType &azimuth,
    const RadialVelErrorEvaluator::DopplerInType &doppler_meas) {
  return RadialVelErrorEvaluator::MakeShared(velocity, bias, azimuth, doppler_meas);
}

}  // namespace se2
}  // namespace steam