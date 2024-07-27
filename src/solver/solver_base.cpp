#include "steam/solver/solver_base.hpp"

#include <iostream>

#include "steam/common/Timer.hpp"

namespace steam {

SolverBase::SolverBase(Problem& problem, const Params& params)
    : problem_(problem),
      state_vector_(problem.getStateVector()),
      params_(params) {
  //
  state_vector_backup_ = state_vector_.lock()->clone();
  // Set current cost from initial problem
  curr_cost_ = prev_cost_ = problem_.cost();
}

void SolverBase::optimize() {
  // Timer
  steam::Timer timer;
  // Optimization loop
  while (!solver_converged_) iterate();
  // Log
  if (params_.verbose)
    std::cout << "Total Optimization Time: " << timer.milliseconds() << " ms"
              << std::endl;
}

void SolverBase::iterate() {
  // Check is solver has already converged
  if (solver_converged_) {
    std::cout << "[STEAM WARN] Requested an iteration when solver has already "
                 "converged, iteration ignored.";
    return;
  }

  // Log on first iteration
  if (params_.verbose && curr_iteration_ == 0) {
    // clang-format off
    std::cout << "Begin Optimization" << std::endl;
    std::cout << "------------------" << std::endl;
    std::cout << "Number of States: " << state_vector_.lock()->getNumberOfStates() << std::endl;
    std::cout << "Number of Cost Terms: " << problem_.getNumberOfCostTerms() << std::endl;
    std::cout << "Initial Cost: " << curr_cost_ << std::endl;
    // clang-format on
  }

  // Update iteration number
  curr_iteration_++;

  // Record previous iteration cost
  prev_cost_ = curr_cost_;

  // Perform an iteration of the implemented solver-type
  double grad_norm = 0.0;
  bool step_success = linearizeSolveAndUpdate(curr_cost_, grad_norm);

  // Check termination criteria
  if (!step_success && fabs(grad_norm) < 1e-6) {
    term_ = TERMINATE_CONVERGED_ZERO_GRADIENT;
    solver_converged_ = true;
  } else if (!step_success) {
    term_ = TERMINATE_STEP_UNSUCCESSFUL;
    solver_converged_ = true;
    throw unsuccessful_step(
        "The steam solver terminated due to being unable to produce a "
        "'successful' step. If this occurs, it is likely that your problem "
        "is very nonlinear and poorly initialized, or is using incorrect "
        "analytical Jacobians. Grad norm was " + std::to_string(fabs(grad_norm)));
  } else if (curr_iteration_ >= params_.max_iterations) {
    term_ = TERMINATE_MAX_ITERATIONS;
    solver_converged_ = true;
  } else if (curr_cost_ <= params_.absolute_cost_threshold) {
    term_ = TERMINATE_CONVERGED_ABSOLUTE_ERROR;
    solver_converged_ = true;
  } else if (fabs(prev_cost_ - curr_cost_) <=
             params_.absolute_cost_change_threshold) {
    term_ = TERMINATE_CONVERGED_ABSOLUTE_CHANGE;
    solver_converged_ = true;
  } else if (fabs(prev_cost_ - curr_cost_) / prev_cost_ <=
             params_.relative_cost_change_threshold) {
    term_ = TERMINATE_CONVERGED_RELATIVE_CHANGE;
    solver_converged_ = true;
  }
  // else if (curr_cost_ > prev_cost_) {
  //   term_ = TERMINATE_COST_INCREASED;
  //   solver_converged_ = true;
  // }

  // Log on final iteration
  if (params_.verbose && solver_converged_)
    std::cout << "Termination Cause: " << term_ << std::endl;
}

double SolverBase::proposeUpdate(const Eigen::VectorXd& perturbation) {
  // Check that an update is not already pending
  if (pending_proposed_state_) {
    throw std::runtime_error(
        "There is already a pending update, accept "
        "or reject before proposing a new one.");
  }
  const auto state_vector = state_vector_.lock();
  if (!state_vector) throw std::runtime_error{"state vector expired"};
  // Make copy of state vector
  state_vector_backup_.copyValues(*(state_vector));
  // Update copy with perturbation
  state_vector->update(perturbation);
  pending_proposed_state_ = true;
  // Test new cost
  return problem_.cost();
}

void SolverBase::acceptProposedState() {
  // Check that an update has been proposed
  if (!pending_proposed_state_)
    throw std::runtime_error("You must call proposeUpdate before accept.");
  // Switch flag, accepting the update
  pending_proposed_state_ = false;
}

void SolverBase::rejectProposedState() {
  // Check that an update has been proposed
  if (!pending_proposed_state_)
    throw std::runtime_error("You must call proposeUpdate before rejecting.");
  // Revert to previous state
  state_vector_.lock()->copyValues(state_vector_backup_);
  // Switch flag, ready for new proposal
  pending_proposed_state_ = false;
}

std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T) {
  switch (T) {
    case SolverBase::TERMINATE_NOT_YET_TERMINATED:
      out << "NOT YET TERMINATED";
      break;
    case SolverBase::TERMINATE_STEP_UNSUCCESSFUL:
      out << "STEP UNSUCCESSFUL";
      break;
    case SolverBase::TERMINATE_MAX_ITERATIONS:
      out << "MAX ITERATIONS";
      break;
    case SolverBase::TERMINATE_CONVERGED_ABSOLUTE_ERROR:
      out << "CONVERGED ABSOLUTE ERROR";
      break;
    case SolverBase::TERMINATE_CONVERGED_ABSOLUTE_CHANGE:
      out << "CONVERGED ABSOLUTE CHANGE";
      break;
    case SolverBase::TERMINATE_CONVERGED_RELATIVE_CHANGE:
      out << "CONVERGED RELATIVE CHANGE";
      break;
    case SolverBase::TERMINATE_CONVERGED_ZERO_GRADIENT:
      out << "CONVERGED GRADIENT IS ZERO";
      break;
    case SolverBase::TERMINATE_COST_INCREASED:
      out << "COST INCREASED";
      break;
  }
  return out;
}

}  // namespace steam
