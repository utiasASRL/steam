//////////////////////////////////////////////////////////////////////////////////////////////
/// \file JacobianTreeLeafNode.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/jacobian/JacobianTreeLeafNode.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
JacobianTreeLeafNode::JacobianTreeLeafNode(const StateVariableBase::ConstPtr& state) : state_(state) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Traverse the Jacobian tree and calculate the Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
void JacobianTreeLeafNode::append(std::vector<Jacobian>* outJacobians) const {

  // Add Jacobian (identity)
  const unsigned int dim = state_->getPerturbDim();
  outJacobians->push_back(Jacobian(state_->getKey(), Eigen::MatrixXd::Identity(dim,dim)));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Traverse the Jacobian tree and calculate the Jacobians, pre-multiplied by lhs
//////////////////////////////////////////////////////////////////////////////////////////////
void JacobianTreeLeafNode::append(const Eigen::MatrixXd& lhs,
                                  std::vector<Jacobian>* outJacobians) const {

  // Check that dimensions match
  if (lhs.cols() != state_->getPerturbDim()) {
    throw std::runtime_error("Jacobian tree had dimension mismatch.");
  }

  // Add Jacobian
  outJacobians->push_back(Jacobian(state_->getKey(), lhs));
}

} // steam