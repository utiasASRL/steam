#pragma once

#include <Eigen/Core>

#include "steam/blockmat/BlockMatrix.hpp"
#include "steam/blockmat/BlockSparseMatrix.hpp"
#include "steam/blockmat/BlockVector.hpp"

#include "steam/solver/solver_base.hpp"

namespace steam {

/**
 * \brief Reports that the decomposition failed. This is due to poor
 * conditioning, or possibly that the matrix is not positive definite.
 */
class decomp_failure : public solver_failure {
 public:
  decomp_failure(const std::string& s) : solver_failure(s) {}
};

class GaussNewtonSolver : public SolverBase {
 public:
  struct Params : public SolverBase::Params {
    ///
    bool reuse_previous_pattern = true;
  };

  GaussNewtonSolver(Problem& problem, const Params& params);

 protected:
  /** \brief Solve the Gauss-Newton system of equations: A*x = b */
  Eigen::VectorXd solveGaussNewton(
      const Eigen::SparseMatrix<double>& approximate_hessian,
      const Eigen::VectorXd& gradient_vector);

 private:
  using SolverType =
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>;
  std::shared_ptr<SolverType> solver() { return hessian_solver_; }

  bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

  /** \brief The solver stored over iterations to reuse the same pattern */
  std::shared_ptr<SolverType> hessian_solver_ = std::make_shared<SolverType>();

  /** \brief Whether the pattern of the approx. Hessian has been analyzed */
  bool pattern_initialized_ = false;

  const Params params_;

  friend class Covariance;  // access the internal solver object
};

}  // namespace steam
