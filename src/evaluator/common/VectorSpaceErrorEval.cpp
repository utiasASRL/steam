//////////////////////////////////////////////////////////////////////////////////////////////
/// \file VectorSpaceErrorEval.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/common/VectorSpaceErrorEval.hpp>

#include <steam/evaluator/jacobian/JacobianTreeBranchNode.hpp>
#include <steam/evaluator/jacobian/JacobianTreeLeafNode.hpp>

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
Eigen::VectorXd VectorSpaceErrorEval::evaluate(std::vector<Jacobian>* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Construct Jacobian
  if(!stateVec_->isLocked()) {
    jacs->resize(1);
    Jacobian& jacref = jacs->back();
    jacref.key = stateVec_->getKey();
    const unsigned int dim = stateVec_->getPerturbDim();
    jacref.jac = -Eigen::MatrixXd::Identity(dim,dim);
  }

  // Return error
  return measurement_ - stateVec_->getValue();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error and relevant Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
std::pair<Eigen::VectorXd, JacobianTreeNode::ConstPtr> VectorSpaceErrorEval::evaluateJacobians() const {

  // Init Jacobian node (null)
  JacobianTreeBranchNode::Ptr jacobianNode;

  // Check if evaluator is active
  if (!stateVec_->isLocked()) {

    // Make Jacobian node
    jacobianNode = JacobianTreeBranchNode::Ptr(new JacobianTreeBranchNode(1));

    // Make leaf node for Landmark
    JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(stateVec_));

    // Add Jacobian
    const unsigned int dim = stateVec_->getPerturbDim();
//    jacobianNode->add(-Eigen::MatrixXd::Identity(dim,dim), leafNode);
    jacobianNode->add(leafNode) = -Eigen::MatrixXd::Identity(dim,dim);
  }

  // Return error
  return std::make_pair(measurement_ - stateVec_->getValue(), jacobianNode);
}

} // steam
