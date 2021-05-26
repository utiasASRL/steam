//////////////////////////////////////////////////////////////////////////////////////////////
/// \file LogMapEvaluator.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/blockauto/transform/LogMapEvaluator.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
LogMapEvaluator::LogMapEvaluator(const TransformEvaluator::ConstPtr& transform) : transform_(transform) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
LogMapEvaluator::Ptr LogMapEvaluator::MakeShared(const TransformEvaluator::ConstPtr& transform) {
  return LogMapEvaluator::Ptr(new LogMapEvaluator(transform));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool LogMapEvaluator::isActive() const {
  return transform_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds references (shared pointers) to active state variables to the map output
//////////////////////////////////////////////////////////////////////////////////////////////
void LogMapEvaluator::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {
  transform_->getActiveStateVariables(outStates);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant 6x1 vector belonging to the se(3) algebra
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> LogMapEvaluator::evaluate() const {
  return transform_->evaluate().vec();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant 6x1 vector belonging to the se(3) algebra and
///        sub-tree of evaluations
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef STEAM_USE_OBJECT_POOL
EvalTreeNode<Eigen::Matrix<double,6,1> >* LogMapEvaluator::evaluateTree() const {

  // Evaluate sub-trees
  EvalTreeNode<lgmath::se3::Transformation>* transform = transform_->evaluateTree();

  // Make new root node -- note we get memory from the pool
  EvalTreeNode<Eigen::Matrix<double,6,1> >* root = EvalTreeNode<Eigen::Matrix<double,6,1> >::pool.getObj();
#else
EvalTreeNode<Eigen::Matrix<double, 6, 1> >::Ptr LogMapEvaluator::evaluateTree() const {

  // Evaluate sub-trees
  EvalTreeNode<lgmath::se3::Transformation>::Ptr transform = transform_->evaluateTree();

  // Make new root node
  auto root = std::make_shared<EvalTreeNode<Eigen::Matrix<double, 6, 1>>>();
#endif
  root->setValue(transform->getValue().vec());

  // Add children
  root->addChild(transform);

  // Return new root node
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Implementation for Block Automatic Differentiation
//////////////////////////////////////////////////////////////////////////////////////////////
template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
void LogMapEvaluator::appendJacobiansImpl(
    const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
#ifdef STEAM_USE_OBJECT_POOL
    EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
#else
    EvalTreeNode<Eigen::Matrix<double,6,1> >::Ptr evaluationTree,
#endif
    std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const {

  // Check if transform is active
  if (transform_->isActive()) {
    Eigen::Matrix<double,LHS_DIM,INNER_DIM> newLhs = lhs * lgmath::se3::vec2jacinv(evaluationTree->getValue());
    transform_->appendBlockAutomaticJacobians(newLhs,
#ifdef STEAM_USE_OBJECT_POOL
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
#else
      std::static_pointer_cast<EvalTreeNode<lgmath::se3::Transformation>>(evaluationTree->childAt(0)),
#endif
      outJacobians);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void LogMapEvaluator::appendBlockAutomaticJacobians(const Eigen::MatrixXd& lhs,
#ifdef STEAM_USE_OBJECT_POOL
                                  EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
#else
                                  EvalTreeNode<Eigen::Matrix<double,6,1> >::Ptr evaluationTree,
#endif
                                  std::vector<Jacobian<> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void LogMapEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,1,6>& lhs,
#ifdef STEAM_USE_OBJECT_POOL
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
#else
                              EvalTreeNode<Eigen::Matrix<double,6,1> >::Ptr evaluationTree,
#endif
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void LogMapEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,2,6>& lhs,
#ifdef STEAM_USE_OBJECT_POOL
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
#else
                              EvalTreeNode<Eigen::Matrix<double,6,1> >::Ptr evaluationTree,
#endif
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void LogMapEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,3,6>& lhs,
#ifdef STEAM_USE_OBJECT_POOL
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
#else
                              EvalTreeNode<Eigen::Matrix<double,6,1> >::Ptr evaluationTree,
#endif
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void LogMapEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,4,6>& lhs,
#ifdef STEAM_USE_OBJECT_POOL
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
#else
                              EvalTreeNode<Eigen::Matrix<double,6,1> >::Ptr evaluationTree,
#endif
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void LogMapEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,6,6>& lhs,
#ifdef STEAM_USE_OBJECT_POOL
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
#else
                              EvalTreeNode<Eigen::Matrix<double,6,1> >::Ptr evaluationTree,
#endif
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

} // se3
} // steam
