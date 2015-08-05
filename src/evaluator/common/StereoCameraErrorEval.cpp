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
Eigen::VectorXd StereoCameraErrorEval::evaluate(const Eigen::MatrixXd& lhs, std::vector<Jacobian>* jacs) const {

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
  eval_->appendJacobians(-lhs*cameraModelJacobian(point_in_c), evaluationTree, jacs);

  // Cleanup tree memory
  delete evaluationTree;

  // Return evaluation
  return meas_ - cameraModel(point_in_c);
}

////////////////////////////////////////////////////////////////////////////////////////////////
///// \brief Evaluate the 4-d measurement error (ul vl ur vr) and sub-tree of evaluations
////////////////////////////////////////////////////////////////////////////////////////////////
//EvalTreeNode<Eigen::VectorXd>* StereoCameraErrorEval::evaluateTree() const {

//  // Evaluate sub-trees
//  EvalTreeNode<Eigen::Vector4d>* point_in_c = eval_->evaluateTree();

//  // Make new root node
//  EvalTreeNode<Eigen::VectorXd>* root =
//      new EvalTreeNode<Eigen::VectorXd>(meas_ - cameraModel(point_in_c->getValue()));

//  // Add children
//  root->addChild(point_in_c);

//  // Return new root node
//  return root;
//}

////////////////////////////////////////////////////////////////////////////////////////////////
///// \brief Evaluate the Jacobian tree
////////////////////////////////////////////////////////////////////////////////////////////////
//void StereoCameraErrorEval::appendJacobians(const Eigen::MatrixXd& lhs,
//                                  EvalTreeNode<Eigen::VectorXd>* evaluationTree,
//                                  std::vector<Jacobian>* outJacobians) const {

//  EvalTreeNode<Eigen::Vector4d>* point_in_c =
//      static_cast<EvalTreeNode<Eigen::Vector4d>*>(evaluationTree->childAt(0));

//  // Check if transform1 is active
//  if (eval_->isActive()) {
//    eval_->appendJacobians(-lhs*cameraModelJacobian(point_in_c->getValue()), point_in_c, outJacobians);
//  }
//}

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
