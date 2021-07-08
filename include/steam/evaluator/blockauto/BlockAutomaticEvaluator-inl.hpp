//////////////////////////////////////////////////////////////////////////////////////////////
/// \file BlockAutomaticEvaluator-inl.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/blockauto/BlockAutomaticEvaluator.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE, int INNER_DIM, int MAX_STATE_SIZE>
BlockAutomaticEvaluator<TYPE,INNER_DIM,MAX_STATE_SIZE>::BlockAutomaticEvaluator() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief General evaluation and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE, int INNER_DIM, int MAX_STATE_SIZE>
TYPE BlockAutomaticEvaluator<TYPE,INNER_DIM,MAX_STATE_SIZE>::evaluate(
  const Eigen::MatrixXd& lhs, std::vector<Jacobian<> >* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation tree - note that this pointer belongs to a pool
#ifdef STEAM_USE_OBJECT_POOL
  EvalTreeNode<TYPE>* tree = this->evaluateTree();
#else
  typename EvalTreeNode<TYPE>::Ptr tree = this->evaluateTree();
#endif

  // Get Jacobians
  this->appendBlockAutomaticJacobians(lhs, tree, jacs);

  // Get evaluation from tree
  TYPE eval = tree->getValue();

#ifdef STEAM_USE_OBJECT_POOL
  // Return tree memory to pool
  EvalTreeNode<TYPE>::pool.returnObj(tree);
#endif

  // Return evaluation
  return eval;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Safe block-automatic evaluation method.
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE, int INNER_DIM, int MAX_STATE_SIZE>
EvalTreeHandle<TYPE> BlockAutomaticEvaluator<TYPE,INNER_DIM,MAX_STATE_SIZE>::getBlockAutomaticEvaluation() const {
  return EvalTreeHandle<TYPE>(this->evaluateTree());
}

} // steam
