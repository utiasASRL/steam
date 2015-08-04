//////////////////////////////////////////////////////////////////////////////////////////////
/// \file JacobianTreeBranchNode.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/jacobian/JacobianTreeBranchNode.hpp>

#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
JacobianTreeBranchNode::JacobianTreeBranchNode(unsigned int reserveNum) {
  children_.reserve(reserveNum);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a child node to the branch
//////////////////////////////////////////////////////////////////////////////////////////////
//void JacobianTreeBranchNode::add(const Eigen::MatrixXd& lhsJacobian,
//                                 const JacobianTreeNode::ConstPtr& child) {

//  // Check for nullptr
//  if (child) {

//    // Add Jacobian edge to list of children
//    children_.push_back(JacobianEdge_t(lhsJacobian, child));
//  } else {

//    // Provided pointer was null
//    throw std::invalid_argument("Tried to add nullptr to JacobianTreeBranchNode");
//  }
//}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a child node to the branch
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd& JacobianTreeBranchNode::add(const JacobianTreeNode::ConstPtr& child) {

  // Check for nullptr
  if (child) {

    // Add Jacobian edge to list of children
    children_.push_back(JacobianEdge_t());
    children_.back().second = child;
    return children_.back().first;
  } else {

    // Provided pointer was null
    throw std::invalid_argument("Tried to add nullptr to JacobianTreeBranchNode");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Traverse the Jacobian tree and calculate the Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
void JacobianTreeBranchNode::append(std::vector<Jacobian>* outJacobians) const {

  // Traverse tree by calling append on children nodes
  for (std::vector<JacobianEdge_t>::const_iterator it = children_.begin();
       it != children_.end(); ++it) {

    // Get hint index in case we need to call merge
    unsigned int hintIndex = outJacobians->size();

    // Note that the new left-hand-side matrix is 'lhs' multiplied by the edge matrix (it->first)
    it->second->append(it->first, outJacobians);

    // If this branch had more than one child, merge any Jacobians
    // that are both with respect to the same state variable
    if (it != children_.begin()) {
      JacobianTreeNode::merge(outJacobians, hintIndex);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Traverse the Jacobian tree and calculate the Jacobians, pre-multiplied by lhs
//////////////////////////////////////////////////////////////////////////////////////////////
void JacobianTreeBranchNode::append(const Eigen::MatrixXd& lhs,
                                    std::vector<Jacobian>* outJacobians) const {

  // Traverse tree by calling append on children nodes
  for (std::vector<JacobianEdge_t>::const_iterator it = children_.begin();
       it != children_.end(); ++it) {

    // Check that dimensions match
    if (lhs.cols() != it->first.rows()) {
      throw std::runtime_error("Jacobian tree had dimension mismatch.");
    }

    // Get hint index in case we need to call merge
    unsigned int hintIndex = outJacobians->size();

    // Note that the new left-hand-side matrix is 'lhs' multiplied by the edge matrix (it->first)
    it->second->append(lhs * it->first, outJacobians);

    // If this branch had more than one child, merge any Jacobians
    // that are both with respect to the same state variable
    if (it != children_.begin()) {
      JacobianTreeNode::merge(outJacobians, hintIndex);
    }
  }
}

} // steam
