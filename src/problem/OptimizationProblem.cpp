//////////////////////////////////////////////////////////////////////////////////////////////
/// \file OptimizationProblem.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/OptimizationProblem.hpp>

#include <iomanip>
#include <steam/common/Timer.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
OptimizationProblem::OptimizationProblem() : firstBackup_(true), pendingProposedState_(false) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add an 'active' state variable
//////////////////////////////////////////////////////////////////////////////////////////////
void OptimizationProblem::addStateVariable(const StateVariableBase::Ptr& state)
{
  stateVec_.addStateVariable(state);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a cost term (should depend on active states that were added to the problem)
//////////////////////////////////////////////////////////////////////////////////////////////
void OptimizationProblem::addCostTerm(const CostTermX::ConstPtr& costTerm) {
  dynamicCostTerms_.add(costTerm);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a collection of cost terms. This function is typically reserved for adding
///        custom collections which contain fixed-size cost terms and evaluators. For general
///        dynamic-size cost terms, use addCostTerm(). Note that the cost terms should depend
///        on active states that were added to the problem.
//////////////////////////////////////////////////////////////////////////////////////////////
void OptimizationProblem::addCostTermCollection(const CostTermCollectionBase::ConstPtr& costTermColl) {
  customCostTerms_.push_back(costTermColl);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute the cost from the collection of cost terms
//////////////////////////////////////////////////////////////////////////////////////////////
double OptimizationProblem::cost() const {

  double cost = 0;

  // Add cost of the default dynamic cost terms
  if (dynamicCostTerms_.size() > 0) {
    cost += dynamicCostTerms_.cost();
  }

  // Add cost of the custom cost-term collections
  for (unsigned int c = 0; c < customCostTerms_.size(); c++) {
    if (customCostTerms_[c]->size() > 0) {
      cost += customCostTerms_[c]->cost();
    }
  }

  return cost;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fill in the supplied block matrices
//////////////////////////////////////////////////////////////////////////////////////////////
void OptimizationProblem::buildGaussNewtonTerms(Eigen::SparseMatrix<double>* approximateHessian,
                                                Eigen::VectorXd* gradientVector) const {
//  double time1 = 0;
//  double time2 = 0;
//  double time3 = 0;

  // Setup Matrices
  //Timer timer;
  std::vector<unsigned int> sqSizes = stateVec_.getStateBlockSizes();
  BlockSparseMatrix A_(sqSizes, true);
  BlockVector b_(sqSizes);
  //time1 = timer.milliseconds(); timer.reset();

  // Add terms from the default dynamic cost terms
  if (dynamicCostTerms_.size() > 0) {
    dynamicCostTerms_.buildGaussNewtonTerms(stateVec_, &A_, &b_);
  }

  // Add terms from the custom cost-term collections
  for (unsigned int c = 0; c < customCostTerms_.size(); c++) {
    if (customCostTerms_[c]->size() > 0) {
      customCostTerms_[c]->buildGaussNewtonTerms(stateVec_, &A_, &b_);
    }
  }

  //time2 = timer.milliseconds(); timer.reset();

  // Convert to Eigen Type - with the block-sparsity pattern
  // ** Note we do not exploit sub-block-sparsity in case it changes at a later iteration
  *approximateHessian = A_.toEigen(false);
  *gradientVector = b_.toEigen();

  //time3 = timer.milliseconds(); timer.reset();

  //std::cout << "top layer: " << time1 << " " << time2 << " " << time3 << " total: " << time1+time2+time3 << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get a reference to the state vector
//////////////////////////////////////////////////////////////////////////////////////////////
const StateVector& OptimizationProblem::getStateVector() const {
  return stateVec_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the total number of cost terms
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int OptimizationProblem::getNumberOfCostTerms() const {

  unsigned int size = 0;

  // Add number of the default dynamic cost terms
  size += dynamicCostTerms_.size();

  // Add number from the custom cost-term collections
  for (unsigned int c = 0; c < customCostTerms_.size(); c++) {
    size += customCostTerms_[c]->size();
  }

  return size;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Propose an update to the state vector.
//////////////////////////////////////////////////////////////////////////////////////////////
double OptimizationProblem::proposeUpdate(const Eigen::VectorXd& stateStep) {

  // Check that an update is not already pending 
  if (pendingProposedState_) {
    throw std::runtime_error("There is already a pending update, accept "
                             "or reject before proposing a new one.");
  }

  // Make copy of state vector
  if (firstBackup_) {
    stateVectorBackup_ = stateVec_;
    firstBackup_ = false;
  } else {
    stateVectorBackup_.copyValues(stateVec_);
  }

  // Update copy with perturbation
  stateVec_.update(stateStep);
  pendingProposedState_ = true;

  // Test new cost
  return this->cost();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Confirm the proposed state update
//////////////////////////////////////////////////////////////////////////////////////////////
void OptimizationProblem::acceptProposedState() {

  // Check that an update has been proposed
  if (!pendingProposedState_) {
    throw std::runtime_error("You must call proposeUpdate before accept.");
  }

  // Switch flag, accepting the update
  pendingProposedState_ = false;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Reject the proposed state update and revert to the previous values
//////////////////////////////////////////////////////////////////////////////////////////////
void OptimizationProblem::rejectProposedState() {

  // Check that an update has been proposed
  if (!pendingProposedState_) {
    throw std::runtime_error("You must call proposeUpdate before rejecting.");
  }

  // Revert to previous state
  stateVec_.copyValues(stateVectorBackup_);

  // Switch flag, ready for new proposal
  pendingProposedState_ = false;
}



} // steam
