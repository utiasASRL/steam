//////////////////////////////////////////////////////////////////////////////////////////////
/// \file StereoCameraErrorEval.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/common/StereoCameraErrorEval.hpp>

#include <steam/evaluator/TransformEvalOperations.hpp>
#include <steam/evaluator/jacobian/JacobianTreeBranchNode.hpp>
#include <steam/evaluator/jacobian/JacobianTreeLeafNode.hpp>

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
Eigen::VectorXd StereoCameraErrorEval::evaluate() const {

  // Return error (between measurement and point estimate projected in camera frame)
  return meas_ - cameraModel(eval_->evaluate());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr) and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd StereoCameraErrorEval::evaluate(std::vector<Jacobian>* jacs) const {

  // Get Jacobians involved in point transformation
  std::vector<Jacobian> jacsTemp;
  Eigen::Vector4d point_in_c = eval_->evaluate(&jacsTemp);

  // Get Jacobian for the camera model
  Eigen::Matrix4d cameraJac = cameraModelJacobian(point_in_c);

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();
  jacs->resize(jacsTemp.size());

  // Calculate all full Jacobians
  for (unsigned int j = 0; j < jacsTemp.size(); j++) {
    Jacobian& jacref = jacs->at(j);
    jacref.key = jacsTemp[j].key;
    jacref.jac = -cameraJac * jacsTemp[j].jac;
  }

  // Return error (between measurement and point estimate projected in camera frame)
  return meas_ - cameraModel(point_in_c);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr) and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
std::pair<Eigen::VectorXd, JacobianTreeNode::ConstPtr> StereoCameraErrorEval::evaluateJacobians() const {

  // Evaluate
  std::pair<Eigen::Vector4d, JacobianTreeNode::ConstPtr> point_in_c = eval_->evaluateJacobians();

  // Get Jacobian for the camera model
  Eigen::Matrix4d cameraJac = cameraModelJacobian(point_in_c.first);

  // Init Jacobian node (null)
  JacobianTreeBranchNode::Ptr jacobianNode;

  // Check if evaluator is active
  if (this->isActive()) {

    // Make Jacobian node
    jacobianNode = JacobianTreeBranchNode::Ptr(new JacobianTreeBranchNode(1));

    // Add Jacobian
    //jacobianNode->add(-cameraJac, point_in_c.second);
    jacobianNode->add(point_in_c.second) = -cameraJac;
  }

  // Return error
  return std::make_pair(meas_ - cameraModel(point_in_c.first), jacobianNode);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Camera model
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d StereoCameraErrorEval::cameraModel(const Eigen::Vector4d& point) const {

  // Precompute values
  double one_over_z = 1.0/point[2];

  // Project point into camera coordinates
  Eigen::Vector4d projectedMeas;
  projectedMeas << intrinsics_->fu *  point[0] * one_over_z           + intrinsics_->cu,
                   intrinsics_->fv *  point[1] * one_over_z           + intrinsics_->cv,
                   intrinsics_->fu * (point[0] - intrinsics_->b) * one_over_z + intrinsics_->cu,
                   intrinsics_->fv *  point[1] * one_over_z           + intrinsics_->cv;
  return projectedMeas;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Camera model Jacobian
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4d StereoCameraErrorEval::cameraModelJacobian(const Eigen::Vector4d& point) const {

  // Precompute values
  double one_over_z = 1.0/point[2];
  double one_over_z2 = one_over_z*one_over_z;

  // Construct Jacobian with respect to x, y, z, and scalar 1.0
  Eigen::Matrix4d jac;
  jac << intrinsics_->fu*one_over_z, 0.0, -intrinsics_->fu *  point[0] * one_over_z2, 0.0,
         0.0, intrinsics_->fv*one_over_z, -intrinsics_->fv *  point[1] * one_over_z2, 0.0,
         intrinsics_->fu*one_over_z, 0.0, -intrinsics_->fu * (point[0] - intrinsics_->b) * one_over_z2, 0.0,
         0.0, intrinsics_->fv*one_over_z, -intrinsics_->fv *  point[1] * one_over_z2, 0.0;
  return jac;
}

} // steam
