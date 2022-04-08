#include "steam/evaluable/stereo/stereo_error_evaluator.hpp"

namespace steam {
namespace stereo {

auto StereoErrorEvaluator::MakeShared(
    const Eigen::Vector4d& meas, const CameraIntrinsics::ConstPtr& intrinsics,
    const Evaluable<PoseInType>::ConstPtr& T_cam_landmark,
    const Evaluable<LmInType>::ConstPtr& landmark) -> Ptr {
  return std::make_shared<StereoErrorEvaluator>(meas, intrinsics,
                                                T_cam_landmark, landmark);
}

StereoErrorEvaluator::StereoErrorEvaluator(
    const Eigen::Vector4d& meas, const CameraIntrinsics::ConstPtr& intrinsics,
    const Evaluable<PoseInType>::ConstPtr& T_cam_landmark,
    const Evaluable<LmInType>::ConstPtr& landmark)
    : meas_(meas),
      intrinsics_(intrinsics),
      eval_(compose(T_cam_landmark, landmark)) {}

bool StereoErrorEvaluator::active() const { return eval_->active(); }

auto StereoErrorEvaluator::value() const -> OutType {
  return meas_ - cameraModel(eval_->value());
}

auto StereoErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  // error between measurement and point estimate projected in camera frame
  const auto child = eval_->forward();
  const auto value = meas_ - cameraModel(child->value());
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void StereoErrorEvaluator::backward(const Eigen::MatrixXd& lhs,
                                    const Node<OutType>::Ptr& node,
                                    Jacobians& jacs) const {
  if (eval_->active()) {
    const auto child = std::static_pointer_cast<Node<LmInType>>(node->at(0));
    // Get Jacobians
    Eigen::Matrix4d new_lhs = (-1) * lhs * cameraModelJacobian(child->value());

    eval_->backward(new_lhs, child, jacs);
  }
}

Eigen::Vector4d StereoErrorEvaluator::cameraModel(
    const Eigen::Vector4d& point) const {
  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics_->b;
  const double one_over_z = 1.0 / z;

  // Project point into camera coordinates
  Eigen::Vector4d projectedMeas;
  // clang-format off
  projectedMeas << intrinsics_->fu * x * one_over_z + intrinsics_->cu,
                   intrinsics_->fv * y * one_over_z + intrinsics_->cv,
                   intrinsics_->fu * xr * one_over_z + intrinsics_->cu,
                   intrinsics_->fv * y * one_over_z + intrinsics_->cv;
  // clang-format on
  return projectedMeas;
}

Eigen::Matrix4d StereoErrorEvaluator::cameraModelJacobian(
    const Eigen::Vector4d& point) const {
  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics_->b;
  const double one_over_z = 1.0 / z;
  const double one_over_z2 = one_over_z * one_over_z;

  // Construct Jacobian with respect to x, y, z, and scalar w
  const double dw = -intrinsics_->fu * intrinsics_->b * one_over_z;
  Eigen::Matrix4d jac;
  // clang-format off
  jac << intrinsics_->fu * one_over_z, 0.0, -intrinsics_->fu * x * one_over_z2, 0.0,
         0.0, intrinsics_->fv * one_over_z, -intrinsics_->fv * y * one_over_z2, 0.0,
         intrinsics_->fu * one_over_z, 0.0, -intrinsics_->fu * xr * one_over_z2, dw,
         0.0, intrinsics_->fv * one_over_z, -intrinsics_->fv * y * one_over_z2, 0.0;
  // clang-format on
  return jac;
}

}  // end namespace stereo
}  // namespace steam
