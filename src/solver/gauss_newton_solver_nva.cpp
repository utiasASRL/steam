#include "steam/solver/gauss_newton_solver_nva.hpp"

#include <iomanip>
#include <iostream>

#include <Eigen/Cholesky>

#include "steam/common/Timer.hpp"

namespace steam {

GaussNewtonSolverNVA::GaussNewtonSolverNVA(Problem& problem,
                                           const Params& params)
    : GaussNewtonSolver(problem, params), params_(params) {}

bool GaussNewtonSolverNVA::linearizeSolveAndUpdate(double& cost,
                                                   double& grad_norm) {
  steam::Timer iter_timer;
  steam::Timer timer;
  double build_time = 0;
  double solve_time = 0;
  double update_time = 0;

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

  if (params_.line_search) {
    const double expected_delta_cost = 0.5 * gradient_vector.transpose() * perturbation;
    if (expected_delta_cost < 0.0) {
      throw std::runtime_error("Expected delta cost must be >= 0.0");
    }
    if (expected_delta_cost < 1.0e-5 || fabs(expected_delta_cost / cost) < 1.0e-7) {
      solver_converged_ = true;
      term_ = TERMINATE_EXPECTED_DELTA_COST_CONVERGED;
    } else {
      double alpha = 1.0;
      for (uint j = 0; j < 3; ++j) {
        timer.reset();
        // Apply update
        cost = proposeUpdate(alpha * perturbation);
        update_time += timer.milliseconds();  
        if (params_.verbose) std::cout << "line search it: " << j << " prev_cost: " << prev_cost_ << " new_cost: " << cost << " alpha: " << alpha << std::endl;
        if (cost <= prev_cost_) {
          acceptProposedState();
          break;
        } else {
          cost = prev_cost_;
          rejectProposedState();
        }
        alpha *= 0.5;
      }
    }
  } else {
    // Apply update
    timer.reset();
    cost = proposeUpdate(perturbation);
    acceptProposedState();
    update_time = timer.milliseconds();
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
              << std::endl;
    // clang-format on
  }

  return true;
}

Eigen::VectorXd GaussNewtonSolverNVA::solveGaussNewton(
    const Eigen::SparseMatrix<double>& approximate_hessian,
    const Eigen::VectorXd& gradient_vector) {
  // Perform a Cholesky factorization of the approximate Hessian matrix
  // Check if the pattern has been initialized
  if (!pattern_initialized_) {
    // The first time we are solving the problem we need to analyze the sparsity
    // pattern
    // ** Note we use approximate-minimal-degree (AMD) reordering.
    //    Also, this step does not actually use the numerical values in
    //    gaussNewtonLHS
    hessian_solver_->analyzePattern(approximate_hessian);
    if (params_.reuse_previous_pattern) pattern_initialized_ = true;
  }

  // Perform a Cholesky factorization of the approximate Hessian matrix
  hessian_solver_->factorize(approximate_hessian);

  // Check if the factorization succeeded
  if (hessian_solver_->info() != Eigen::Success) {
    throw decomp_failure(
        "During steam solve, Eigen LLT decomposition failed. "
        "It is possible that the matrix was ill-conditioned, in which case "
        "adding a prior may help. On the other hand, it is also possible that "
        "the problem you've constructed is not positive semi-definite.");
  }

  // todo - it would be nice to check the condition number (not just the
  // determinant) of the solved system... need to find a fast way to do this

  // Do the backward pass, using the Cholesky factorization (fast)
  return hessian_solver_->solve(gradient_vector);
}

}  // namespace steam
