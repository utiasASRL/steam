#include "steam/solver2/dogleg_gauss_newton_solver.hpp"

#include <iomanip>
#include <iostream>

#include "steam/common/Timer.hpp"

namespace steam {

DoglegGaussNewtonSolver2::DoglegGaussNewtonSolver2(Problem& problem,
                                                   const Params& params)
    : GaussNewtonSolver(problem, params), params_(params) {}

bool DoglegGaussNewtonSolver2::linearizeSolveAndUpdate(double& cost,
                                                       double& grad_norm) {
  steam::Timer iter_timer;
  steam::Timer timer;
  double build_time = 0;
  double solve_time = 0;
  double update_time = 0;
  double actual_to_predicted_ratio;
  std::string dogleg_segment;
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

  // Solve system
  timer.reset();

  // Get gradient descent step
  Eigen::VectorXd grad_descent_step =
      getCauchyPoint(approximate_hessian, gradient_vector);
  double grad_descent_norm = grad_descent_step.norm();

  Eigen::VectorXd gauss_newton_step =
      solveGaussNewton(approximate_hessian, gradient_vector);
  double gauss_newton_norm = gauss_newton_step.norm();

  solve_time = timer.milliseconds();

  // Apply update (w line search)
  timer.reset();

  // Initialize trust region size (if first time)
  if (trust_region_size_ == 0.0) trust_region_size_ = gauss_newton_norm;

  // Perform dogleg step
  unsigned int num_backtrack = 0;
  Eigen::VectorXd dogleg_step;
  Eigen::VectorXd gd2gn_vector;
  for (; num_backtrack < params_.max_shrink_steps; num_backtrack++) {
    // Get step
    if (gauss_newton_norm <= trust_region_size_) {
      // Trust region larger than Gauss Newton step
      dogleg_step = gauss_newton_step;
      dogleg_segment = "Gauss Newton";
    } else if (grad_descent_norm >= trust_region_size_) {
      // Trust region smaller than Gradient Descent step (Cauchy point)
      dogleg_step =
          (trust_region_size_ / grad_descent_norm) * grad_descent_step;
      dogleg_segment = "Grad Descent";
    } else {
      // Trust region lies between the GD and GN steps, use interpolation
      if (gauss_newton_step.rows() != grad_descent_step.rows()) {
        throw std::logic_error(
            "Gauss-Newton and gradient descent dimensions did not match.");
      }

      // Get interpolation direction
      gd2gn_vector = gauss_newton_step - grad_descent_step;

      // Calculate interpolation constant
      double gd_dot_gd2gn = grad_descent_step.transpose() * gd2gn_vector;
      double gd2gn_sqnorm = gd2gn_vector.squaredNorm();
      double interp_const =
          (-gd_dot_gd2gn + std::sqrt(gd_dot_gd2gn * gd_dot_gd2gn +
                                     (trust_region_size_ * trust_region_size_ -
                                      grad_descent_norm * grad_descent_norm) *
                                         gd2gn_sqnorm)) /
          gd2gn_sqnorm;

      // Interpolate step
      dogleg_step = grad_descent_step + interp_const * gd2gn_vector;
      dogleg_segment = "Interp GN&GD";
    }

    // Calculate the predicted reduction; note that a positive value denotes a
    // reduction in cost
    double proposed_cost = proposeUpdate(dogleg_step);
    double actual_reduc = prev_cost_ - proposed_cost;
    double predicted_reduc =
        predictedReduction(approximate_hessian, gradient_vector, dogleg_step);
    actual_to_predicted_ratio = actual_reduc / predicted_reduc;

    // Check ratio of predicted reduction to actual reduction achieved
    if (actual_to_predicted_ratio > params_.ratio_threshold_shrink) {
      // Good enough ratio to accept proposed state
      acceptProposedState();
      cost = proposed_cost;
      if (actual_to_predicted_ratio > params_.ratio_threshold_grow) {
        // Ratio is strong enough to increase trust region size
        // Note: we take the max below, so that if the trust region is already
        // much larger
        //   than the steps we are taking, we do not grow it unnecessarily
        trust_region_size_ = std::max(trust_region_size_,
                                      params_.grow_coeff * dogleg_step.norm());
      }
      break;
    } else {
      // Cost did not reduce enough, or possibly increased,
      // reject proposed state and reduce the size of the trust region
      rejectProposedState();  // Restore old state vector
      // Reduce step size (backtrack)
      trust_region_size_ *= params_.shrink_coeff;
      num_tr_decreases++;  // Count number of shrinks for logging
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
                << std::right << std::setw(11) << std::setfill(' ') << "TR shrink"
                << std::right << std::setw(11) << std::setfill(' ') << "AvP Ratio"
                << std::right << std::setw(16) << std::setfill(' ') << "dogleg segment"
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
              << std::right << std::setw(16) << std::setfill(' ') << dogleg_segment
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

Eigen::VectorXd DoglegGaussNewtonSolver2::getCauchyPoint(
    const Eigen::SparseMatrix<double>& approximate_hessian,
    const Eigen::VectorXd& gradient_vector) {
  double num = gradient_vector.squaredNorm();
  double den =
      gradient_vector.transpose() *
      (approximate_hessian.selfadjointView<Eigen::Upper>() * gradient_vector);
  return (num / den) * gradient_vector;
}

double DoglegGaussNewtonSolver2::predictedReduction(
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
