//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformErrorEval.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/common/TransformErrorEval.hpp>

#include <steam/evaluator/TransformEvalOperations.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor - error is difference between 'T' and identity (in Lie algebra space)
//////////////////////////////////////////////////////////////////////////////////////////////
TransformErrorEval::TransformErrorEval(const se3::TransformEvaluator::ConstPtr& T) {
  errorEvaluator_ = se3::tran2vec(T);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convenience constructor - error between meas_T_21 and T_21
//////////////////////////////////////////////////////////////////////////////////////////////
TransformErrorEval::TransformErrorEval(const lgmath::se3::Transformation& meas_T_21,
                                       const se3::TransformEvaluator::ConstPtr& T_21) {

  // Construct the evaluator using the convenient transform evaluators
  se3::FixedTransformEvaluator::ConstPtr meas = se3::FixedTransformEvaluator::MakeShared(meas_T_21);
  errorEvaluator_ = se3::tran2vec(se3::compose(meas, se3::inverse(T_21)));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convenience constructor - error between meas_T_21 and T_20*inv(T_10)
//////////////////////////////////////////////////////////////////////////////////////////////
TransformErrorEval::TransformErrorEval(const lgmath::se3::Transformation& meas_T_21,
                                       const se3::TransformStateVar::ConstPtr& T_20,
                                       const se3::TransformStateVar::ConstPtr& T_10) {

  // Construct the evaluator using the convenient transform evaluators
  se3::FixedTransformEvaluator::ConstPtr meas = se3::FixedTransformEvaluator::MakeShared(meas_T_21);
  se3::TransformStateEvaluator::ConstPtr t10 = se3::TransformStateEvaluator::MakeShared(T_10);
  se3::TransformStateEvaluator::ConstPtr t20 = se3::TransformStateEvaluator::MakeShared(T_20);
  errorEvaluator_ = se3::tran2vec(se3::compose(se3::compose(meas, t10), se3::inverse(t20)));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool TransformErrorEval::isActive() const {
  return errorEvaluator_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 6-d measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd TransformErrorEval::evaluate() const {
  return errorEvaluator_->evaluate();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 6-d measurement error and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd TransformErrorEval::evaluate(std::vector<Jacobian>* jacs) const {
  return errorEvaluator_->evaluate(jacs);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 6-d measurement error and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
std::pair<Eigen::VectorXd, JacobianTreeNode::ConstPtr> TransformErrorEval::evaluateJacobians() const {

  //return errorEvaluator_->evaluateJacobians();
  std::pair<Eigen::Matrix<double,6,1>, JacobianTreeNode::ConstPtr> error = errorEvaluator_->evaluateJacobians();
  return std::make_pair(error.first, error.second);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 6-d measurement error and sub-tree of evaluations
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<Eigen::VectorXd>* TransformErrorEval::evaluateTree() const {

  // Evaluate sub-trees
  EvalTreeNode<Eigen::Matrix<double,6,1> >* error = errorEvaluator_->evaluateTree();

  // Make new root node
  EvalTreeNode<Eigen::VectorXd>* root =
      new EvalTreeNode<Eigen::VectorXd>(error->getValue());

  // Add children
  root->addChild(error);

  // Return new root node
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void TransformErrorEval::appendJacobians(const Eigen::MatrixXd& lhs,
                                  EvalTreeNode<Eigen::VectorXd>* evaluationTree,
                                  std::vector<Jacobian>* outJacobians) const {

  // Check if transform1 is active
  if (errorEvaluator_->isActive()) {
    errorEvaluator_->appendJacobians(lhs,
      static_cast<EvalTreeNode<Eigen::Matrix<double,6,1> >*>(evaluationTree->childAt(0)),
      outJacobians);
  }
}

} // steam
