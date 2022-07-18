#include "steam/solver2/lev_marq_gauss_newton_solver.hpp"

#include <iomanip>
#include <iostream>

#include "steam/common/Timer.hpp"

namespace steam {

LevMarqGaussNewtonSolver2::LevMarqGaussNewtonSolver2(Problem& problem,
                                                     const Params& params)
    : GaussNewtonSolver(problem, params), params_(params) {}

bool LevMarqGaussNewtonSolver2::linearizeSolveAndUpdate(double& cost,
                                                        double& grad_norm) {
  steam::Timer iter_timer;
  steam::Timer timer;
  double build_time = 0;
  double solve_time = 0;
  double update_time = 0;
  double actual_to_predicted_ratio = 0;
  unsigned int num_tr_decreases = 0;

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

  // Perform LM Search
  unsigned int num_backtrack = 0;
  for (; num_backtrack < params_.max_shrink_steps; num_backtrack++) {
    // Solve system
    timer.reset();
    bool decomp_success = true;
    Eigen::VectorXd lev_marq_step;
    try {
      lev_marq_step =
          solveLevMarq(approximate_hessian, gradient_vector, diag_coeff_);
    } catch (const decomp_failure2& e) {
      decomp_success = false;
    }
    solve_time += timer.milliseconds();

    // Test new cost
    timer.reset();

    // If decomposition was successful, calculate step quality
    double proposed_cost = 0;
    if (decomp_success) {
      // Calculate the predicted reduction; note that a positive value denotes a
      // reduction in cost
      proposed_cost = proposeUpdate(lev_marq_step);
      double actual_reduc = prev_cost_ - proposed_cost;
      double predicted_reduc = predictedReduction(
          approximate_hessian, gradient_vector, lev_marq_step);
      actual_to_predicted_ratio = actual_reduc / predicted_reduc;
    }

    // Check ratio of predicted reduction to actual reduction achieved
    if (actual_to_predicted_ratio > params_.ratio_threshold && decomp_success) {
      // Good enough ratio to accept proposed state
      acceptProposedState();
      cost = proposed_cost;
      // move towards gauss newton
      diag_coeff_ = std::max(diag_coeff_ * params_.shrink_coeff, 1e-7);

      // Timing
      update_time += timer.milliseconds();
      break;
    } else {
      // Cost did not reduce enough, possibly increased, or decomposition
      // failed. Reject proposed state and reduce the size of the trust region
      if (decomp_success) {
        // Restore old state vector
        rejectProposedState();
      }
      // Move towards gradient descent
      diag_coeff_ = std::min(diag_coeff_ * params_.grow_coeff, 1e7);
      num_tr_decreases++;  // Count number of shrinks for logging

      // Timing
      update_time += timer.milliseconds();
    }
  }

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
                << std::right << std::setw(11) << std::setfill(' ') << "TR shrink"
                << std::right << std::setw(11) << std::setfill(' ') << "AvP Ratio"
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
              << std::right << std::setw(11) << std::setfill(' ') << num_tr_decreases
              << std::right << std::setw(11) << std::setfill(' ') << std::setprecision(3) << std::fixed << actual_to_predicted_ratio << std::resetiosflags(std::ios::fixed)
              << std::endl;
    // clang-format on
  }

  // Return successfulness
  if (num_backtrack < params_.max_shrink_steps) {
    return true;
  } else {
    return false;
  }
}

Eigen::VectorXd LevMarqGaussNewtonSolver2::solveLevMarq(
    Eigen::SparseMatrix<double>& approximate_hessian,
    const Eigen::VectorXd& gradient_vector, double diagonal_coeff) {
  // Augment diagonal of the 'hessian' matrix
  for (int i = 0; i < approximate_hessian.outerSize(); i++) {
    approximate_hessian.coeffRef(i, i) *= (1.0 + diagonal_coeff);
  }

  // Solve system
  Eigen::VectorXd lev_marq_step;
  try {
    // Solve for the LM step using the Cholesky factorization (fast)
    lev_marq_step = solveGaussNewton(approximate_hessian, gradient_vector);
  } catch (const decomp_failure2& ex) {
    // Revert diagonal of the 'hessian' matrix
    for (int i = 0; i < approximate_hessian.outerSize(); i++) {
      approximate_hessian.coeffRef(i, i) /= (1.0 + diagonal_coeff);
    }
    // Throw up again
    throw ex;
  }

  // Revert diagonal of the 'hessian' matrix
  for (int i = 0; i < approximate_hessian.outerSize(); i++) {
    approximate_hessian.coeffRef(i, i) /= (1.0 + diagonal_coeff);
  }

  return lev_marq_step;
}

double LevMarqGaussNewtonSolver2::predictedReduction(
    const Eigen::SparseMatrix<double>& approximate_hessian,
    const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step) {
  // grad^T * step - 0.5 * step^T * Hessian * step
  double grad_tran_step = gradient_vector.transpose() * step;
  double step_trans_hessian_step =
      step.transpose() *
      (approximate_hessian.selfadjointView<Eigen::Upper>() * step);
  return grad_tran_step - 0.5 * step_trans_hessian_step;
}

}  // namespace steam
