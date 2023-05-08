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

void StereoErrorEvaluator::getRelatedVarKeys(KeySet& keys) const {
  eval_->getRelatedVarKeys(keys);
}

auto StereoErrorEvaluator::value() const -> OutType {
  return meas_ - cameraModel(intrinsics_, eval_->value());
}

auto StereoErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  // error between measurement and point estimate projected in camera frame
  const auto child = eval_->forward();
  const auto value = meas_ - cameraModel(intrinsics_, child->value());
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
    Eigen::Matrix4d new_lhs = (-1) * lhs * cameraModelJacobian(intrinsics_, child->value());

    eval_->backward(new_lhs, child, jacs);
  }
}

Eigen::Vector4d cameraModel(
    const CameraIntrinsics::ConstPtr& intrinsics, const Eigen::Vector4d& point) {
  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics->b;
  const double one_over_z = 1.0 / z;

  // Project point into camera coordinates
  Eigen::Vector4d projectedMeas;
  // clang-format off
  projectedMeas << intrinsics->fu * x * one_over_z + intrinsics->cu,
                   intrinsics->fv * y * one_over_z + intrinsics->cv,
                   intrinsics->fu * xr * one_over_z + intrinsics->cu,
                   intrinsics->fv * y * one_over_z + intrinsics->cv;
  // clang-format on
  return projectedMeas;
}

Eigen::Matrix4d cameraModelJacobian(
    const CameraIntrinsics::ConstPtr& intrinsics, const Eigen::Vector4d& point) {
  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics->b;
  const double one_over_z = 1.0 / z;
  const double one_over_z2 = one_over_z * one_over_z;

  // Construct Jacobian with respect to x, y, z, and scalar w
  const double dw = -intrinsics->fu * intrinsics->b * one_over_z;
  Eigen::Matrix4d jac;
  // clang-format off
  jac << intrinsics->fu * one_over_z, 0.0, -intrinsics->fu * x * one_over_z2, 0.0,
         0.0, intrinsics->fv * one_over_z, -intrinsics->fv * y * one_over_z2, 0.0,
         intrinsics->fu * one_over_z, 0.0, -intrinsics->fu * xr * one_over_z2, dw,
         0.0, intrinsics->fv * one_over_z, -intrinsics->fv * y * one_over_z2, 0.0;
  // clang-format on
  return jac;
}


LandmarkNoiseEvaluator::LandmarkNoiseEvaluator(const Eigen::Vector4d& landmark_mean,
                             const Eigen::Matrix3d& landmark_cov,
                             const Eigen::Matrix4d& meas_noise,
                             const CameraIntrinsics::ConstPtr& intrinsics,
                             const se3::SE3StateVar::ConstPtr& T_query_map) : 
intrinsics_(intrinsics),
meas_noise_(meas_noise),
mean_(landmark_mean),
T_query_map_(T_query_map) {
  // compute the dialated phi;
  dialated_phi_.setZero();
  dialated_phi_.block(0,0,3,3) = landmark_cov;
  if(!positiveDefinite<3>(landmark_cov)) {
    std::cout <<  "\nmigrated cov is bad!!!\n";
  }
}

// bool LandmarkNoiseEvaluator::active() {
//   return true;
// }

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief evaluatecovariance
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,4,4> LandmarkNoiseEvaluator::value() {
  // TODO: Check to see if we need to recaulculate (add a change flag to steam variables.)

  // Add the measurement noise.
  last_computed_cov_ = meas_noise_;

  if(!positiveDefinite<4>(meas_noise_)) {
    std::cout << "measurement noise is bad!!";
  }
  // evaluate the steam transform evaluator
  const lgmath::se3::Transformation T_l_p = T_query_map_->evaluate();

  // Compute the new landmark noise
  Eigen::Matrix<double,4,4> lm_noise_l = T_l_p.matrix() * dialated_phi_ * T_l_p.matrix().transpose();

  Eigen::Matrix<double,4,3> dialation_matrix;
  dialation_matrix.setZero();
  dialation_matrix.block(0,0,3,3) = Eigen::Matrix<double,3,3>::Identity();
  Eigen::Matrix<double,3,3> lm_noise_l_3 = dialation_matrix.transpose() * lm_noise_l * dialation_matrix;

  if(positiveDefinite<3>(lm_noise_l_3)) {
    // compute the camera model jacobian based on the transformed mean.
    camera_jacobian_j_ = stereo::cameraModelJacobian(intrinsics_, T_l_p*mean_);

    Eigen::Matrix<double,4,4> lm_noise = camera_jacobian_j_ * lm_noise_l * camera_jacobian_j_.transpose();
    Eigen::Matrix<double,3,3> lm_noise_3 = dialation_matrix.transpose() * lm_noise * dialation_matrix;

    if(positiveDefinite<3>(lm_noise_3)) {
      last_computed_cov_ += lm_noise;
    } else {
      std::cout << "\nmigrated noise is not positive definite!!\n";
    }
    // return the new noise.
  } else {
    std::cout << "\nlm_noise_l is bad!!\n";
  }

  if (!positiveDefinite<4>(last_computed_cov_)) {
    std::cout << "sum of noise is bad...";
  }
  return last_computed_cov_;
}


}  // end namespace stereo
}  // namespace steam
