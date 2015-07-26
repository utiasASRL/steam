//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SolverBase.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/solver/SolverBase.hpp>

#include <iostream>

#include <steam/common/Timer.hpp>

namespace steam {


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SolverBase::SolverBase(OptimizationProblem* problem) : problem_(problem),
    currIteration_(0), solverConverged_(false), term_(TERMINATE_NOT_YET_TERMINATED) {
  currCost_ = prevCost_ = problem_->cost();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not the solver is converged
//////////////////////////////////////////////////////////////////////////////////////////////
bool SolverBase::converged() const {
  return solverConverged_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Perform an iteration of the solver
//////////////////////////////////////////////////////////////////////////////////////////////
void SolverBase::iterate() {

  // Check is solver has already converged
  if (solverConverged_) {
    std::cout << "[STEAM WARN] Requested an interation when solver has already converged, iteration ignored.";
    return;
  }

  // Log on first iteration
  if (this->getSolverBaseParams().verbose && currIteration_ == 0) {
    std::cout << "Begin Optimization" << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << "Number of States: " << problem_->getStateVector().getNumberOfStates() << std::endl;
    std::cout << "Number of Cost Terms: " << problem_->getCostTerms().size() << std::endl;
    std::cout << "Initial Cost: " << currCost_ << std::endl;
  }

  // Update iteration number
  currIteration_++;

  // Record previous iteration cost
  prevCost_ = currCost_;

  // Perform an iteration of the implemented solver-type
  bool stepSuccess = linearizeSolveAndUpdate(&currCost_);

  // Check termination criteria
  if (!stepSuccess) {
    term_ = TERMINATE_STEP_UNSUCCESSFUL;
    solverConverged_ = true;
  } else if (currIteration_ >= this->getSolverBaseParams().maxIterations) {
    term_ = TERMINATE_MAX_ITERATIONS;
    solverConverged_ = true;
  } else if ( currCost_ <= this->getSolverBaseParams().absoluteCostThreshold ) {
    term_ = TERMINATE_CONVERGED_ABSOLUTE_ERROR;
    solverConverged_ = true;
  } else if ( fabs(prevCost_ - currCost_) <= this->getSolverBaseParams().absoluteCostChangeThreshold ) {
    term_ = TERMINATE_CONVERGED_ABSOLUTE_CHANGE;
    solverConverged_ = true;
  } else if ( fabs(prevCost_ - currCost_)/prevCost_ <= this->getSolverBaseParams().relativeCostChangeThreshold ) {
    term_ = TERMINATE_CONVERGED_RELATIVE_CHANGE;
    solverConverged_ = true;
  }

  // Log on final iteration
  if (this->getSolverBaseParams().verbose && solverConverged_) {
    std::cout << "Termination Cause: " << term_ << std::endl;
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Perform iterations until convergence
///        This function is made to be simple and require no private methods so that users
///        can choose to control the loop themselves.
//////////////////////////////////////////////////////////////////////////////////////////////
void SolverBase::optimize() {
  // Timer
  steam::Timer timer;

  // Optimization loop
  while(!this->converged()) {
    this->iterate();
  }

  // Log
  if (this->getSolverBaseParams().verbose) {
    std::cout << "Total Optimization Time: " << timer.milliseconds() << " ms" << std::endl;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Return termination cause
//////////////////////////////////////////////////////////////////////////////////////////////
SolverBase::Termination SolverBase::getTerminationCause() {
  return term_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Return current iteration number
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int SolverBase::getCurrIteration() const {
  return currIteration_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Return previous iteration cost evaluation
//////////////////////////////////////////////////////////////////////////////////////////////
double SolverBase::getPrevCost() const {
  return prevCost_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get reference to optimization problem
//////////////////////////////////////////////////////////////////////////////////////////////
OptimizationProblem& SolverBase::getProblem() {
  return *problem_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get const reference to optimization problem
//////////////////////////////////////////////////////////////////////////////////////////////
const OptimizationProblem& SolverBase::getProblem() const {
  return *problem_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Print termination cause
//////////////////////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T) {
  switch(T) {
    case SolverBase::TERMINATE_NOT_YET_TERMINATED        : out << "NOT YET TERMINATED"; break;
    case SolverBase::TERMINATE_STEP_UNSUCCESSFUL         : out << "STEP UNSUCCESSFUL"; break;
    case SolverBase::TERMINATE_MAX_ITERATIONS            : out << "MAX ITERATIONS"; break;
    case SolverBase::TERMINATE_CONVERGED_ABSOLUTE_ERROR  : out << "CONVERGED ABSOLUTE ERROR"; break;
    case SolverBase::TERMINATE_CONVERGED_ABSOLUTE_CHANGE : out << "CONVERGED ABSOLUTE CHANGE"; break;
    case SolverBase::TERMINATE_CONVERGED_RELATIVE_CHANGE : out << "CONVERGED RELATIVE CHANGE"; break;
  }
  return out;
}

} // steam
