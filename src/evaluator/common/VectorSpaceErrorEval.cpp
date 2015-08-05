//////////////////////////////////////////////////////////////////////////////////////////////
/// \file VectorSpaceErrorEval.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/common/VectorSpaceErrorEval.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
VectorSpaceErrorEval::VectorSpaceErrorEval(const Eigen::VectorXd& measurement, const VectorSpaceStateVar::ConstPtr& stateVec)
  : measurement_(measurement), stateVec_(stateVec) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool VectorSpaceErrorEval::isActive() const {
  return !stateVec_->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd VectorSpaceErrorEval::evaluate() const {
  return measurement_ - stateVec_->getValue();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error and relevant Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd VectorSpaceErrorEval::evaluate(const Eigen::MatrixXd& lhs, std::vector<Jacobian>* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Check that dimensions match
  if (lhs.cols() != stateVec_->getPerturbDim()) {
    throw std::runtime_error("evaluate had dimension mismatch.");
  }

  // Construct Jacobian
  if(!stateVec_->isLocked()) {
    jacs->resize(1);
    Jacobian& jacref = jacs->back();
    jacref.key = stateVec_->getKey();
    jacref.jac = -lhs;
  }

  // Return error
  return measurement_ - stateVec_->getValue();
}

////////////////////////////////////////////////////////////////////////////////////////////////
///// \brief Evaluate the measurement error and sub-tree
////////////////////////////////////////////////////////////////////////////////////////////////
//EvalTreeNode<Eigen::VectorXd>* VectorSpaceErrorEval::evaluateTree() const {
//  return new EvalTreeNode<Eigen::VectorXd>(measurement_ - stateVec_->getValue());
//}

////////////////////////////////////////////////////////////////////////////////////////////////
///// \brief Evaluate the Jacobian tree
////////////////////////////////////////////////////////////////////////////////////////////////
//void VectorSpaceErrorEval::appendJacobians(const Eigen::MatrixXd& lhs,
//                                  EvalTreeNode<Eigen::VectorXd>* evaluationTree,
//                                  std::vector<Jacobian>* outJacobians) const {

//  // Check if state is locked
//  if (!stateVec_->isLocked()) {

//    // Check that dimensions match
//    if (lhs.cols() != stateVec_->getPerturbDim()) {
//      throw std::runtime_error("appendJacobians had dimension mismatch.");
//    }

//    // Add Jacobian
//    outJacobians->push_back(Jacobian(stateVec_->getKey(), -lhs));
//  }
//}

} // steam
