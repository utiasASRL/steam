//////////////////////////////////////////////////////////////////////////////////////////////
/// \file BlockAutomaticEvaluator-inl.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/BlockAutomaticEvaluator.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalType>
BlockAutomaticEvaluator<EvalType>::BlockAutomaticEvaluator() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General evaluation and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename EvalType>
EvalType BlockAutomaticEvaluator<EvalType>::evaluate(const Eigen::MatrixXd& lhs,
                                                     std::vector<Jacobian>* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation tree
  EvalTreeNode<EvalType>* tree = this->evaluateTree();

  // Get Jacobians
  this->appendJacobians(lhs, tree, jacs);

  // Get evaluation from tree
  EvalType eval = tree->getValue();

  // Cleanup tree memory
  delete tree;

  // Return evaluation
  return eval;
}

} // steam
