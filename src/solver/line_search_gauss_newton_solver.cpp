#include "steam/solver/line_search_gauss_newton_solver.hpp"

#include <iomanip>
#include <iostream>

#include "steam/common/Timer.hpp"

namespace steam {

LineSearchGaussNewtonSolver::LineSearchGaussNewtonSolver(Problem& problem,
                                                         const Params& params)
    : GaussNewtonSolver(problem, params), params_(params) {}

bool LineSearchGaussNewtonSolver::linearizeSolveAndUpdate(double& cost,
                                                          double& grad_norm) {
  steam::Timer iter_timer;
  steam::Timer timer;
  double build_time = 0;
  double solve_time = 0;
  double update_time = 0;

  // Initialize new cost with old cost incase of failure
  cost = prev_cost_;

  // The 'left-hand-side' of the Gauss-Newton problem, generally known as the
  // approximate Hessian matrix (note we only store the upper-triangular
  // elements)
  Eigen::SparseMatrix<double> approximate_hessian;
  // The 'right-hand-side' of the Gauss-Newton problem, generally known as the
  // gradient vector
  Eigen::VectorXd gradient_vector;

  // Construct system of equations
  timer.reset();
  problem_.buildGaussNewtonTerms(approximate_hessian, gradient_vector);
  grad_norm = gradient_vector.norm();
  build_time = timer.milliseconds();

  // Solve system
  timer.reset();
  Eigen::VectorXd perturbation =
      solveGaussNewton(approximate_hessian, gradient_vector);
  solve_time = timer.milliseconds();

  // Apply update (w line search)
  timer.reset();

  // Perform line search
  double backtrack_coeff = 1.0;  // step size multiplier
  unsigned int num_backtrack = 0;
  for (; num_backtrack < params_.max_backtrack_steps; num_backtrack++) {
    // Test new cost
    double proposed_cost = proposeUpdate(backtrack_coeff * perturbation);
    if (proposed_cost <= prev_cost_) {
      // cost went down (or is the same, x = 0)
      acceptProposedState();
      cost = proposed_cost;
      break;
    } else {
      // cost went up
      rejectProposedState();  // restore old state vector
      // reduce step size (backtrack)
      backtrack_coeff = params_.backtrack_multiplier * backtrack_coeff;
    }
  }

  update_time = timer.milliseconds();

  // Print report line if verbose option enabled
  if (params_.verbose) {
    if (curr_iteration_ == 1) {
      // clang-format off
      std::cout << std::right << std::setw( 4) << std::setfill(' ') << "iter"
                << std::right << std::setw(12) << std::setfill(' ') << "cost"
                << std::right << std::setw(12) << std::setfill(' ') << "build (ms)"
                << std::right << std::setw(12) << std::setfill(' ') << "solve (ms)"
                << std::right << std::setw(13) << std::setfill(' ') << "update (ms)"
                << std::right << std::setw(11) << std::setfill(' ') << "time (ms)"
                << std::right << std::setw(13) << std::setfill(' ') << "search_coeff"
                << std::endl;
      // clang-format on
    }
    // clang-format off
    std::cout << std::right << std::setw( 4) << std::setfill(' ') << curr_iteration_
              << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(5) << cost
              << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(3) << std::fixed << build_time << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(12) << std::setfill(' ') << std::setprecision(3) << std::fixed << solve_time << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(13) << std::setfill(' ') << std::setprecision(3) << std::fixed << update_time << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(11) << std::setfill(' ') << std::setprecision(3) << std::fixed << iter_timer.milliseconds() << std::resetiosflags(std::ios::fixed)
              << std::right << std::setw(13) << std::setfill(' ') << std::setprecision(3) << std::fixed << backtrack_coeff << std::resetiosflags(std::ios::fixed)
              << std::endl;
    // clang-format on
  }

  // Return successfulness
  if (num_backtrack < params_.max_backtrack_steps) {
    return true;
  } else {
    return false;
  }
}

}  // namespace steam
