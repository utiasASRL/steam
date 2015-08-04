//////////////////////////////////////////////////////////////////////////////////////////////
/// \file JacobianTreeNode.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/jacobian/JacobianTreeNode.hpp>

#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
JacobianTreeNode::JacobianTreeNode() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Go through vector of Jacobians and check for Jacobians which are with respect to
///        the same state variable, and merge them.
///
/// For efficiency, specify a hintIndex, which specifies that Jacobians before hintIndex
/// cannot be multiples of eachother.
//////////////////////////////////////////////////////////////////////////////////////////////
void JacobianTreeNode::merge(std::vector<Jacobian>* outJacobians, unsigned int hintIndex) {

  // Check inputs
  if (hintIndex > outJacobians->size()) {
    throw std::invalid_argument("The specified hintIndex is beyond the size of outJacobians");
  }

  // Iterate over the 'safe' non-duplicate Jacobians
  for (unsigned int j = 0; j < hintIndex; j++) {

    // Iterate over the branched (other) Jacobians
    for (unsigned int k = hintIndex; k < outJacobians->size();) {

      // Check if Jacobian j and k are w.r.t the same state variable.
      // If so, we must merge them and erase the second entry
      if (outJacobians->at(j).key.equals(outJacobians->at(k).key)) {

        // Merge
        outJacobians->at(j).jac += outJacobians->at(k).jac;

        // Erase duplicate
        outJacobians->erase(outJacobians->begin() + k);

        // Assuming merge has been called consistently, there should not exist
        // more than one duplicate in second half.. so we can go to the next 'j'
        break;

      } else {
        k++;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the Jacobians with respect to leaf state variables
//////////////////////////////////////////////////////////////////////////////////////////////
std::vector<Jacobian> JacobianTreeNode::getJacobians() const {
  std::vector<Jacobian> jacobians;
  this->append(&jacobians);
  return jacobians;
}

} // steam
