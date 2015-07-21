//////////////////////////////////////////////////////////////////////////////////////////////
/// \file LevMarqGaussNewtonSolver.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/solver/LevMarqGaussNewtonSolver.hpp>

#include <glog/logging.h>
#include <iostream>

#include <steam/common/Timer.hpp>


namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
LevMarqGaussNewtonSolver::LevMarqGaussNewtonSolver(OptimizationProblem* problem, const Params& params)
  : GaussNewtonSolverBase(problem), params_(params) {

  // a small diagonal coefficient means that it is closer to using just a gauss newton step
  diagCoeff = 1e-7;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Build the system, solve for a step size and direction, and update the state
//////////////////////////////////////////////////////////////////////////////////////////////
bool LevMarqGaussNewtonSolver::linearizeSolveAndUpdate(double* newCost) {
  CHECK_NOTNULL(newCost);

  // Logging variables
  steam::Timer iterTimer;
  steam::Timer timer;
  double buildTime = 0;
  double solveTime = 0;
  double updateTime = 0;
  double actualToPredictedRatio;
  unsigned int numTrDecreases = 0;

  // Initialize new cost with old cost incase of failure
  *newCost = this->getPrevCost();

  // Construct system of equations
  timer.reset();
  this->buildGaussNewtonTerms();
  buildTime = timer.milliseconds();

  // Perform LM Search
  unsigned int nBacktrack = 0;
  for (; nBacktrack < params_.maxShrinkSteps; nBacktrack++) {

    // Solve system
    timer.reset();
    Eigen::VectorXd levMarqStep = this->solveGaussNewtonForLM(diagCoeff);
    solveTime += timer.milliseconds();

    // Test new cost
    timer.reset();
    double proposedCost = this->getProblem().proposeUpdate(levMarqStep);
    double actualReduc = this->getPrevCost() - proposedCost;   // a reduction in cost is positive
    double predictedReduc = this->predictedReduction(levMarqStep); // a reduction in cost is positive
    actualToPredictedRatio = actualReduc/predictedReduc;

    // Check ratio of predicted reduction to actual reduction achieved
    if (actualToPredictedRatio > params_.ratioThreshold) {
      // Good enough ratio to accept proposed state
      this->getProblem().acceptProposedState();
      *newCost = proposedCost;
      diagCoeff = std::max(diagCoeff*params_.shrinkCoeff, 1e-7); // move towards gauss newton
      break;
    } else {
      // Cost did not reduce enough, or possibly increased,
      // reject proposed state and reduce the size of the trust region
      this->getProblem().rejectProposedState(); // Restore old state vector
      diagCoeff = std::min(diagCoeff*params_.growCoeff, 1e7); // move towards gradient descent
      numTrDecreases++; // Count number of shrinks for logging
    }
    updateTime += timer.milliseconds();
  }

  // Print report line if verbose option enabled
  if (params_.verbose) {
    if (this->getCurrIteration() == 1) {
        std::cout  << std::right << std::setw( 4) << std::setfill(' ') << "iter"
                   << std::right << std::setw(12) << std::setfill(' ') << "cost"
                   << std::right << std::setw(12) << std::setfill(' ') << "build (ms)"
                   << std::right << std::setw(12) << std::setfill(' ') << "solve (ms)"
                   << std::right << std::setw(13) << std::setfill(' ') << "update (ms)"
                   << std::right << std::setw(11) << std::setfill(' ') << "time (ms)"
                   << std::right << std::setw(11) << std::setfill(' ') << "TR shrink"
                   << std::right << std::setw(11) << std::setfill(' ') << "AvP Ratio"
                   << std::endl;
    }
    std::cout << std::right << std::setw(4)  << std::setfill(' ') << this->getCurrIteration()
              << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(5) << *newCost
              << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(3) << std::fixed << buildTime << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(3) << std::fixed << solveTime << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(13) << std::setfill(' ') << std::setprecision(3) << std::fixed << updateTime << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(11) << std::setfill(' ') << std::setprecision(3) << std::fixed << iterTimer.milliseconds() << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(11) << std::setfill(' ') << numTrDecreases
              << std::right << std::setw(11) << std::setfill(' ') << std::setprecision(3) << std::fixed << actualToPredictedRatio << std::resetiosflags(std::ios::fixed)
              << std::endl;
  }

  // Return successfulness
  if (nBacktrack < params_.maxShrinkSteps) {
    return true;
  } else {
    return false;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Casts parameters to base type (for SolverBase class)
//////////////////////////////////////////////////////////////////////////////////////////////
const SolverBase::Params& LevMarqGaussNewtonSolver::getSolverBaseParams() const {
  return params_;
}

} // steam

