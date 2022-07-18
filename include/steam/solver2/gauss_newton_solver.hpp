#pragma once

#include <Eigen/Core>

#include "steam/blockmat/BlockMatrix.hpp"
#include "steam/blockmat/BlockSparseMatrix.hpp"
#include "steam/blockmat/BlockVector.hpp"

#include "steam/solver2/solver_base.hpp"

namespace steam {

/**
 * \brief Reports that the decomposition failed. This is due to poor
 * conditioning, or possibly that the matrix is not positive definite.
 */
class decomp_failure2 : public solver_failure2 {
 public:
  decomp_failure2(const std::string& s) : solver_failure2(s) {}
};

class GaussNewtonSolver : public SolverBase2 {
 public:
  GaussNewtonSolver(Problem& problem, const Params& params);

 protected:
  /** \brief Solve the Gauss-Newton system of equations: A*x = b */
  Eigen::VectorXd solveGaussNewton(
      const Eigen::SparseMatrix<double>& approximate_hessian,
      const Eigen::VectorXd& gradient_vector);

 private:
  bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

  /** \brief The solver stored over iterations to reuse the same pattern */
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>
      hessian_solver_;
  /** \brief Whether the pattern of the approx. Hessian has been analyzed */
  bool pattern_initialized_ = false;

  const Params params_;
};

}  // namespace steam
