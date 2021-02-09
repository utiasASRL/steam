//////////////////////////////////////////////////////////////////////////////////////////////
/// \file ConstAccTransformEvaluator.cpp
///
/// \author Tim Tang, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/blockauto/transform/ConstAccTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ConstAccTransformEvaluator::ConstAccTransformEvaluator(
    const VectorSpaceStateVar::Ptr& velocity, 
    const VectorSpaceStateVar::Ptr& acceleration, 
    const Time& time) : velocity_(velocity), acceleration_(acceleration),
  time_(time) {

  if(velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("[ConstVelTransformEval] velocity was not 6D.");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
ConstAccTransformEvaluator::Ptr ConstAccTransformEvaluator::MakeShared(
    const VectorSpaceStateVar::Ptr& velocity, 
    const VectorSpaceStateVar::Ptr& acceleration,
    const Time& time) {
  return ConstAccTransformEvaluator::Ptr(new ConstAccTransformEvaluator(velocity, 
    acceleration, time));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool ConstAccTransformEvaluator::isActive() const {
  return !velocity_->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds references (shared pointers) to active state variables to the map output
//////////////////////////////////////////////////////////////////////////////////////////////
void ConstAccTransformEvaluator::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {
  if (this->isActive()) {
    (*outStates)[velocity_->getKey().getID()] = velocity_;
    (*outStates)[acceleration_->getKey().getID()] = acceleration_;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation ConstAccTransformEvaluator::evaluate() const {
  Eigen::Matrix<double,6,1> xi = time_.seconds() * velocity_->getValue() +
  0.5*time_.seconds()*time_.seconds()*acceleration_->getValue();
  return lgmath::se3::Transformation(xi);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* ConstAccTransformEvaluator::evaluateTree() const {

  // Make new leaf node -- note we get memory from the pool
  EvalTreeNode<lgmath::se3::Transformation>* result = EvalTreeNode<lgmath::se3::Transformation>::pool.getObj();
  Eigen::Matrix<double,6,1> xi = time_.seconds() * velocity_->getValue() +
  0.5*time_.seconds()*time_.seconds()*acceleration_->getValue();
  result->setValue(lgmath::se3::Transformation(xi));
  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Implementation for Block Automatic Differentiation
//////////////////////////////////////////////////////////////////////////////////////////////
template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
void ConstAccTransformEvaluator::appendJacobiansImpl(
    const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const {

  if (!velocity_->isLocked()) {

    // Make jacobian
    Eigen::Matrix<double,6,1> xi = time_.seconds() * velocity_->getValue() +
    0.5*time_.seconds()*time_.seconds()*acceleration_->getValue();

    // TODO: check this is correct
    Eigen::Matrix<double,6,6> jac_vel = time_.seconds() * lgmath::se3::vec2jac(xi);
    Eigen::Matrix<double,6,6> jac_acc = 0.5*time_.seconds()*time_.seconds()*lgmath::se3::vec2jac(xi);

    // Add Jacobian
    outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(velocity_->getKey(), lhs*jac_vel));
    outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(acceleration_->getKey(), lhs*jac_acc));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void ConstAccTransformEvaluator::appendBlockAutomaticJacobians(const Eigen::MatrixXd& lhs,
                                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                              std::vector<Jacobian<> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstAccTransformEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstAccTransformEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstAccTransformEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstAccTransformEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

void ConstAccTransformEvaluator::appendBlockAutomaticJacobians(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

} // se3
} // steam
