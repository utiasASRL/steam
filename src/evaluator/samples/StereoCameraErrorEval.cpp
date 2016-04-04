//////////////////////////////////////////////////////////////////////////////////////////////
/// \file StereoCameraErrorEval.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/samples/StereoCameraErrorEval.hpp>

namespace steam {

namespace stereo {

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Camera model Jacobian
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d cameraModelJacobian(const CameraIntrinsics::ConstPtr &intrinsics, const Eigen::Vector4d& point) {
  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics->b;
  const double one_over_z = 1.0/z;
  const double one_over_z2 = one_over_z*one_over_z;

  // Construct Jacobian with respect to x, y, z, and scalar w
  const double dw = -intrinsics->fu * intrinsics->b * one_over_z;
  Eigen::Matrix4d jac;
  jac << intrinsics->fu * one_over_z, 0.0, -intrinsics->fu * x  * one_over_z2, 0.0,
         0.0, intrinsics->fv * one_over_z, -intrinsics->fv * y  * one_over_z2, 0.0,
         intrinsics->fu * one_over_z, 0.0, -intrinsics->fu * xr * one_over_z2,  dw,
         0.0, intrinsics->fv * one_over_z, -intrinsics->fv * y  * one_over_z2, 0.0;
  return jac;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
LandmarkNoiseEvaluator::LandmarkNoiseEvaluator(const Eigen::Vector4d& landmark_mean,
                             const Eigen::Matrix3d& landmark_cov,
                             const Eigen::Matrix4d& meas_noise,
                             const CameraIntrinsics::ConstPtr& intrinsics,
                             const se3::TransformEvaluator::ConstPtr& T_cam_landmark) {
  intrinsics_ = intrinsics;
  dialated_phi_.setZero();
  dialated_phi_.block(0,0,3,3) = landmark_cov;
  meas_noise_ = meas_noise;
  mean_ = landmark_mean;
  T_cam_landmark_ = T_cam_landmark;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// @brief evaluatecovariance
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,4,4> LandmarkNoiseEvaluator::evaluateCovariance() {
  // TODO: Check to see if we need to recaulculate;
  const auto &T_l_p = T_cam_landmark_->evaluate();
  // 2. Calculate G

  camera_jacobian_j_ = stereo::cameraModelJacobian(intrinsics_,T_l_p*mean_);
  auto lm_noise = camera_jacobian_j_ * T_l_p.matrix() * dialated_phi_ * 
                   T_l_p.matrix().transpose() * camera_jacobian_j_.transpose();
  last_computed_cov_ = meas_noise_ + lm_noise;
  return last_computed_cov_;
}

} // end namespace stereo

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
StereoCameraErrorEval::StereoCameraErrorEval(const Eigen::Vector4d& meas,
                                     const stereo::CameraIntrinsics::ConstPtr& intrinsics,
                                     const se3::TransformEvaluator::ConstPtr& T_cam_landmark,
                                     const se3::LandmarkStateVar::Ptr& landmark)
  : meas_(meas), intrinsics_(intrinsics), eval_(se3::compose(T_cam_landmark, landmark)) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool StereoCameraErrorEval::isActive() const {
  return eval_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d StereoCameraErrorEval::evaluate() const {

  // Return error (between measurement and point estimate projected in camera frame)
  return meas_ - cameraModel(eval_->evaluate());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr) and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d StereoCameraErrorEval::evaluate(const Eigen::Matrix4d& lhs, std::vector<Jacobian<4,6> >* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation tree
  EvalTreeHandle<Eigen::Vector4d> blkAutoEvalPointInCameraFrame =
      eval_->getBlockAutomaticEvaluation();

  // Get evaluation from tree
  const Eigen::Vector4d& pointInCamFrame = blkAutoEvalPointInCameraFrame.getValue();

  // Get Jacobians
  Eigen::Matrix4d newLhs = (-1)*lhs*cameraModelJacobian(pointInCamFrame);
  eval_->appendBlockAutomaticJacobians(newLhs, blkAutoEvalPointInCameraFrame.getRoot(), jacs);

  // Return evaluation
  return meas_ - cameraModel(pointInCamFrame);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Camera model
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d StereoCameraErrorEval::cameraModel(const Eigen::Vector4d& point) const {

  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics_->b;
  const double one_over_z = 1.0/z;

  // Project point into camera coordinates
  Eigen::Vector4d projectedMeas;
  projectedMeas << intrinsics_->fu *  x  * one_over_z + intrinsics_->cu,
                   intrinsics_->fv *  y  * one_over_z + intrinsics_->cv,
                   intrinsics_->fu *  xr * one_over_z + intrinsics_->cu,
                   intrinsics_->fv *  y  * one_over_z + intrinsics_->cv;
  return projectedMeas;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Camera model Jacobian
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d StereoCameraErrorEval::cameraModelJacobian(const Eigen::Vector4d& point) const {

  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics_->b;
  const double one_over_z = 1.0/z;
  const double one_over_z2 = one_over_z*one_over_z;

  // Construct Jacobian with respect to x, y, z, and scalar w
  const double dw = -intrinsics_->fu * intrinsics_->b * one_over_z;
  Eigen::Matrix4d jac;
  jac << intrinsics_->fu * one_over_z, 0.0, -intrinsics_->fu * x  * one_over_z2, 0.0,
         0.0, intrinsics_->fv * one_over_z, -intrinsics_->fv * y  * one_over_z2, 0.0,
         intrinsics_->fu * one_over_z, 0.0, -intrinsics_->fu * xr * one_over_z2,  dw,
         0.0, intrinsics_->fv * one_over_z, -intrinsics_->fv * y  * one_over_z2, 0.0;
  return jac;
}

} // steam
