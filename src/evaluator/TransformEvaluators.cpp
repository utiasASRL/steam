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
/// \brief Implementation for Block Automatic Differentiation
//////////////////////////////////////////////////////////////////////////////////////////////
template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
void TransformStateEvaluator::appendJacobiansImpl(
    const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const {
  if (!transform_->isLocked()) {
    outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(transform_->getKey(), lhs));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void TransformStateEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void TransformStateEvaluator::appendJacobians(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void TransformStateEvaluator::appendJacobians(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void TransformStateEvaluator::appendJacobians(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void TransformStateEvaluator::appendJacobians(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void TransformStateEvaluator::appendJacobians(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
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
/// \brief Implementation for Block Automatic Differentiation
//////////////////////////////////////////////////////////////////////////////////////////////
template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
void FixedTransformEvaluator::appendJacobiansImpl(
    const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const {

  // Do nothing
  return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void FixedTransformEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                              std::vector<Jacobian<> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void FixedTransformEvaluator::appendJacobians(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void FixedTransformEvaluator::appendJacobians(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void FixedTransformEvaluator::appendJacobians(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void FixedTransformEvaluator::appendJacobians(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void FixedTransformEvaluator::appendJacobians(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}


// Velocity transform evaluator

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ConstVelTransformEvaluator::ConstVelTransformEvaluator(
    const VectorSpaceStateVar::Ptr& velocity, const Time& time) : velocity_(velocity),
  time_(time) {

  if(velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("[ConstVelTransformEval] velocity was not 6D.");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
ConstVelTransformEvaluator::Ptr ConstVelTransformEvaluator::MakeShared(
    const VectorSpaceStateVar::Ptr& velocity, const Time& time) {
  return ConstVelTransformEvaluator::Ptr(new ConstVelTransformEvaluator(velocity, time));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool ConstVelTransformEvaluator::isActive() const {
  return !velocity_->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds references (shared pointers) to active state variables to the map output
//////////////////////////////////////////////////////////////////////////////////////////////
void ConstVelTransformEvaluator::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {
  if (this->isActive()) {
    (*outStates)[velocity_->getKey().getID()] = velocity_;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation ConstVelTransformEvaluator::evaluate() const {
  Eigen::Matrix<double,6,1> xi = time_.seconds() * velocity_->getValue();
  return lgmath::se3::Transformation(xi);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* ConstVelTransformEvaluator::evaluateTree() const {

  // Make new leaf node -- note we get memory from the pool
  EvalTreeNode<lgmath::se3::Transformation>* result = EvalTreeNode<lgmath::se3::Transformation>::pool.getObj();
  Eigen::Matrix<double,6,1> xi = time_.seconds() * velocity_->getValue();
  result->setValue(lgmath::se3::Transformation(xi));
  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Implementation for Block Automatic Differentiation
//////////////////////////////////////////////////////////////////////////////////////////////
template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
void ConstVelTransformEvaluator::appendJacobiansImpl(
    const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const {

  if (!velocity_->isLocked()) {

    // Make jacobian
    Eigen::Matrix<double,6,1> xi = time_.seconds() * velocity_->getValue();
    Eigen::Matrix<double,6,6> jac = time_.seconds() * lgmath::se3::vec2jac(xi);

    // Add Jacobian
    outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(velocity_->getKey(), lhs*jac));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void ConstVelTransformEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                              std::vector<Jacobian<> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstVelTransformEvaluator::appendJacobians(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstVelTransformEvaluator::appendJacobians(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstVelTransformEvaluator::appendJacobians(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstVelTransformEvaluator::appendJacobians(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstVelTransformEvaluator::appendJacobians(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

} // se3
} // steam
