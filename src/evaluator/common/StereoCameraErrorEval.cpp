//////////////////////////////////////////////////////////////////////////////////////////////
/// \file StereoCameraErrorEval.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/common/StereoCameraErrorEval.hpp>

#include <steam/evaluator/TransformEvalOperations.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
StereoCameraErrorEval::StereoCameraErrorEval(const Eigen::Vector4d& meas,
                                     const CameraIntrinsics::ConstPtr& intrinsics,
                                     const se3::TransformEvaluator::ConstPtr& T_cam_landmark,
                                     const se3::LandmarkStateVar::ConstPtr& landmark)
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
  EvalTreeNode<Eigen::Vector4d>* evaluationTree = eval_->evaluateTree();

  // Get evaluation from tree
  Eigen::Vector4d point_in_c = evaluationTree->getValue();

  // Get Jacobians
  //eval_->appendJacobians4((-1)*lhs*cameraModelJacobian(point_in_c), evaluationTree, jacs);
  Eigen::Matrix4d newLhs = (-1)*lhs*cameraModelJacobian(point_in_c);
  eval_->appendJacobians4(newLhs, evaluationTree, jacs);

  // Return tree memory to pool
  EvalTreeNode<Eigen::Vector4d>::pool.returnObj(evaluationTree);

  // Return evaluation
  return meas_ - cameraModel(point_in_c);
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
  double one_over_z = 1.0/z;
  double one_over_z2 = one_over_z*one_over_z;

  // Construct Jacobian with respect to x, y, z, and scalar 1.0
  Eigen::Matrix4d jac;
  jac << intrinsics_->fu*one_over_z, 0.0, -intrinsics_->fu *  x  * one_over_z2, 0.0,
         0.0, intrinsics_->fv*one_over_z, -intrinsics_->fv *  y  * one_over_z2, 0.0,
         intrinsics_->fu*one_over_z, 0.0, -intrinsics_->fu *  xr * one_over_z2, 0.0,
         0.0, intrinsics_->fv*one_over_z, -intrinsics_->fv *  y  * one_over_z2, 0.0;
  return jac;
}

} // steam
