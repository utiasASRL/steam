//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvaluators.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/TransformEvaluators.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

// State variable

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
TransformStateEvaluator::TransformStateEvaluator(const se3::TransformStateVar::Ptr& transform) : transform_(transform) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
TransformStateEvaluator::Ptr TransformStateEvaluator::MakeShared(const se3::TransformStateVar::Ptr& transform) {
  return TransformStateEvaluator::Ptr(new TransformStateEvaluator(transform));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool TransformStateEvaluator::isActive() const {
  return !transform_->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds references (shared pointers) to active state variables to the map output
//////////////////////////////////////////////////////////////////////////////////////////////
void TransformStateEvaluator::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {
  if (this->isActive()) {
    (*outStates)[transform_->getKey().getID()] = transform_;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation TransformStateEvaluator::evaluate() const {
  return transform_->getValue();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* TransformStateEvaluator::evaluateTree() const {

  // Make new leaf node -- note we get memory from the pool
  EvalTreeNode<lgmath::se3::Transformation>* result = EvalTreeNode<lgmath::se3::Transformation>::pool.getObj();
  result->setValue(transform_->getValue());
  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void TransformStateEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                              std::vector<Jacobian<> >* outJacobians) const {

  // Check if state is locked
  if (!transform_->isLocked()) {

    // Check that dimensions match
    if (lhs.cols() != transform_->getPerturbDim()) {
      throw std::runtime_error("appendJacobians had dimension mismatch.");
    }

    // Add Jacobian
    outJacobians->push_back(Jacobian<>(transform_->getKey(), lhs));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void TransformStateEvaluator::appendJacobians1(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  if (!transform_->isLocked()) {
    outJacobians->push_back(Jacobian<1,6>(transform_->getKey(), lhs));
  }
}

void TransformStateEvaluator::appendJacobians2(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  if (!transform_->isLocked()) {
    outJacobians->push_back(Jacobian<2,6>(transform_->getKey(), lhs));
  }
}

void TransformStateEvaluator::appendJacobians3(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  if (!transform_->isLocked()) {
    outJacobians->push_back(Jacobian<3,6>(transform_->getKey(), lhs));
  }
}

void TransformStateEvaluator::appendJacobians4(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  if (!transform_->isLocked()) {
    outJacobians->push_back(Jacobian<4,6>(transform_->getKey(), lhs));
  }
}

void TransformStateEvaluator::appendJacobians6(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  if (!transform_->isLocked()) {
    outJacobians->push_back(Jacobian<6,6>(transform_->getKey(), lhs));
  }
}

// Fixed value

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
FixedTransformEvaluator::FixedTransformEvaluator(const lgmath::se3::Transformation& transform) : transform_(transform) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
FixedTransformEvaluator::Ptr FixedTransformEvaluator::MakeShared(const lgmath::se3::Transformation& transform) {
  return FixedTransformEvaluator::Ptr(new FixedTransformEvaluator(transform));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool FixedTransformEvaluator::isActive() const {
  return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds references (shared pointers) to active state variables to the map output
//////////////////////////////////////////////////////////////////////////////////////////////
void FixedTransformEvaluator::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation FixedTransformEvaluator::evaluate() const {
  return transform_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* FixedTransformEvaluator::evaluateTree() const {

  // Make new leaf node -- note we get memory from the pool
  EvalTreeNode<lgmath::se3::Transformation>* result = EvalTreeNode<lgmath::se3::Transformation>::pool.getObj();// makeNew();
  result->setValue(transform_);
  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void FixedTransformEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                              std::vector<Jacobian<> >* outJacobians) const {
  // Do nothing
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void FixedTransformEvaluator::appendJacobians1(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  // Do nothing
  return;
}

void FixedTransformEvaluator::appendJacobians2(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  // Do nothing
  return;
}

void FixedTransformEvaluator::appendJacobians3(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  // Do nothing
  return;
}

void FixedTransformEvaluator::appendJacobians4(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  // Do nothing
  return;
}

void FixedTransformEvaluator::appendJacobians6(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  // Do nothing
  return;
}

} // se3
} // steam
