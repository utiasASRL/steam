#pragma once

#include "steam/solver2/gauss_newton_solver.hpp"

namespace steam {

class LevMarqGaussNewtonSolver2 : public GaussNewtonSolver {
 public:
  struct Params : public GaussNewtonSolver::Params {
    /// Minimum ratio of actual to predicted reduction, shrink trust region if
    /// lower, else grow (range: 0.0-1.0)
    double ratio_threshold = 0.25;
    /// Amount to shrink by (range: <1.0)
    double shrink_coeff = 0.1;
    /// Amount to grow by (range: >1.0)
    double grow_coeff = 10.0;
    /// Maximum number of times to shrink trust region before giving up
    unsigned int max_shrink_steps = 50;
  };

  LevMarqGaussNewtonSolver2(Problem& problem, const Params& params);

 private:
  bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

  /**
   * \brief Solve the Levenberg–Marquardt system of equations:
   *        A*x = b, A = (J^T*J + diagonalCoeff*diag(J^T*J))
   */
  Eigen::VectorXd solveLevMarq(Eigen::SparseMatrix<double>& approximate_hessian,
                               const Eigen::VectorXd& gradient_vector,
                               double diagonal_coeff);

  /** \brief Get the predicted cost reduction based on the proposed step */
  double predictedReduction(
      const Eigen::SparseMatrix<double>& approximate_hessian,
      const Eigen::VectorXd& gradient_vector, const Eigen::VectorXd& step);

  /**
   * \brief Diagonal multiplier
   * (lambda in most papers - related to trust region size)
   */
  double diag_coeff_ = 1e-7;

  const Params params_;
};

}  // namespace steam
