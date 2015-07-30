//////////////////////////////////////////////////////////////////////////////////////////////
/// \file GaussNewtonSolverBase.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/solver/GaussNewtonSolverBase.hpp>

#include <iostream>
#include <Eigen/Cholesky>

#include <steam/common/Timer.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
GaussNewtonSolverBase::GaussNewtonSolverBase(OptimizationProblem* problem) : SolverBase(problem) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Build the left-hand and right-hand sides of the Gauss-Newton system of equations
//////////////////////////////////////////////////////////////////////////////////////////////
void GaussNewtonSolverBase::buildGaussNewtonTerms() {

  // Locally disable any internal eigen multithreading -- we do our own OpenMP
  Eigen::setNbThreads(1);

  // Setup Matrices
  std::vector<unsigned int> sqSizes = this->getProblem().getStateVector().getStateBlockSizes();
  BlockSparseMatrix A_(sqSizes, true);
  BlockVector b_(sqSizes);

  // For each cost term
  #pragma omp parallel for num_threads(NUMBER_OF_OPENMP_THREADS)
  for (unsigned int c = 0 ; c < this->getProblem().getCostTerms().size(); c++) {

    // Compute the weighted and whitened errors and jacobians
    // err = sqrt(w)*sqrt(R^-1)*rawError
    // jac = sqrt(w)*sqrt(R^-1)*rawJacobian
    std::vector<Jacobian> jacobians;
    Eigen::VectorXd error = this->getProblem().getCostTerms().at(c)->evalWeightedAndWhitened(&jacobians);

    // For each jacobian
    for (unsigned int i = 0; i < jacobians.size(); i++) {

      // Get the key and state range affected
      unsigned int blkIdx1 = this->getProblem().getStateVector().getStateBlockIndex(jacobians[i].key);

      // Intermediate variable saves time for multiple uses of transpose
      Eigen::MatrixXd j1Transpose = jacobians[i].jac.transpose();

      // Calculate terms needed to update the right-hand-side
      Eigen::MatrixXd b_add = -j1Transpose*error;

      // Update the right-hand side (thread critical)
      #pragma omp critical(b_update)
      {
        b_.add(blkIdx1, b_add);
      }

      // For each jacobian (in upper half)
      for (unsigned int j = i; j < jacobians.size(); j++) {

        // Get the key and state range affected
        unsigned int blkIdx2 = this->getProblem().getStateVector().getStateBlockIndex(jacobians[j].key);

        // Calculate terms needed to update the Gauss-Newton left-hand side
        unsigned int row;
        unsigned int col;
        Eigen::MatrixXd a_add;
        if (blkIdx1 <= blkIdx2) {
          row = blkIdx1;
          col = blkIdx2;
          a_add = j1Transpose*jacobians[j].jac;
        } else {
          row = blkIdx2;
          col = blkIdx1;
          a_add = jacobians[j].jac.transpose()*jacobians[i].jac;
        }

        // Update the left-hand side (thread critical)
        #pragma omp critical(a_update)
        {
          A_.add(row, col, a_add);
        }
      }
    }
  }

  // Convert to Eigen Type
  gaussNewtonLHS = A_.toEigen();
  gaussNewtonRHS = b_.toEigen();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Solve the Gauss-Newton system of equations: A*x = b
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GaussNewtonSolverBase::solveGaussNewton() const {

  // Setup Sparse LLT solver
  // Uses approximate-minimal-degree (AMD) reordering
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;

  // Perform a Cholesky factorization of A (takes the bulk of the time)
  solver.compute(gaussNewtonLHS);
  if (solver.info() != Eigen::Success) {
    throw decomp_failure("During steam solve, Eigen LLT decomposition failed.
                         "It is possible that the matrix was ill-conditioned, in which case "
                         "adding a prior may help. On the other hand, it is also possible that "
                         "the problem you've constructed is not positive semi-definite.");
  }

  // todo, also check for condition number? (not just determinant)

  // Do the backward pass, using the Cholesky factorization (fast)
  return solver.solve(gaussNewtonRHS);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Solve the Levenberg–Marquardt system of equations:
///        A*x = b, A = (J^T*J + diagonalCoeff*diag(J^T*J))
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GaussNewtonSolverBase::solveGaussNewtonForLM(double diagonalCoeff) {

  // Augment diagonal of the 'hessian' matrix
  // when a newer version of eigen is available, the below line should work:
  //   gaussNewtonLHS_A.diagonal() *= (1.0 + diagonalCoeff);
  for (int i = 0; i < gaussNewtonLHS.outerSize(); i++) {
    gaussNewtonLHS.coeffRef(i,i) *= (1.0 + diagonalCoeff);
  }

  // Solve system
  Eigen::VectorXd levMarqStep;
  try {
    levMarqStep = this->solveGaussNewton();
  } catch (const decomp_failure& e) {
    // Revert diagonal of the 'hessian' matrix
    for (int i = 0; i < gaussNewtonLHS.outerSize(); i++) {
      gaussNewtonLHS.coeffRef(i,i) /= (1.0 + diagonalCoeff);
    }
    throw e;
  }

  // Revert diagonal of the 'hessian' matrix
  // when a newer version of eigen is available, the below line should work:
  //   gaussNewtonLHS_A.diagonal() /= (1.0 + diagonalCoeff);
  for (int i = 0; i < gaussNewtonLHS.outerSize(); i++) {
    gaussNewtonLHS.coeffRef(i,i) /= (1.0 + diagonalCoeff);
  }

  return levMarqStep;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Find the Cauchy point (used for the Dogleg method).
///        The cauchy point is the optimal step length in the gradient descent direction.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GaussNewtonSolverBase::getCauchyPoint() const {
  double num = gaussNewtonRHS.squaredNorm();
  double den = gaussNewtonRHS.transpose() * (gaussNewtonLHS.selfadjointView<Eigen::Upper>() * gaussNewtonRHS);
  return (num/den)*gaussNewtonRHS;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the predicted cost reduction based on the proposed step
//////////////////////////////////////////////////////////////////////////////////////////////
double GaussNewtonSolverBase::predictedReduction(const Eigen::VectorXd& step) const {
  // b^T * s - 0.5 * s^T * A * s
  double bts = gaussNewtonRHS.transpose() * step;
  double stAs = step.transpose() * (gaussNewtonLHS.selfadjointView<Eigen::Upper>() * step);
  return bts - 0.5 * stAs;
}

} // steam

