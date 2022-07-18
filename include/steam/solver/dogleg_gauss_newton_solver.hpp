#pragma once

#include "steam/solver/gauss_newton_solver.hpp"

namespace steam {

class DoglegGaussNewtonSolver : public GaussNewtonSolver {
 public:
  struct Params : public GaussNewtonSolver::Params {
    /// Minimum ratio of actual to predicted cost reduction, shrink trust region
    /// if lower (range: 0.0-1.0)
    double ratio_threshold_shrink = 0.25;
    /// Grow trust region if ratio of actual to predicted cost reduction above
    /// this (range: 0.0-1.0)
    double ratio_threshold_grow = 0.75;
    /// Amount to shrink by (range: <1.0)
    double shrink_coeff = 0.5;
    /// Amount to grow by (range: >1.0)
    double grow_coeff = 3.0;
    /// Maximum number of times to shrink trust region before giving up
    unsigned int max_shrink_steps = 50;
  };

  DoglegGaussNewtonSolver(Problem& problem, const Params& params);

 private:
  bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

  /**
   * \brief Find the Cauchy point (used for the Dogleg method).
   * The cauchy point is the optimal step length in the gradient descent
   * direction.
   */
  Eigen::VectorXd getCauchyPoint(
      const Eigen::SparseMatrix<double>& approximate_hessian,
      const Eigen::VectorXd& gradient_vector);

  /** \brief Get the predicted cost reduction based on the proposed step */
  double predictedReduction(
      const Eigen::SparseMatrix<double>& approximate_hessian,
      const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step);

  /** \brief Trust region size */
  double trust_region_size_ = 0.0;

  const Params params_;
};

}  // namespace steam
