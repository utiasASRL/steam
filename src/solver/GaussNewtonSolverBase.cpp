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
GaussNewtonSolverBase::GaussNewtonSolverBase(OptimizationProblem* problem) :
  SolverBase(problem), patternInitialized_(false), factorizedInformationSuccesfully_(false) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Query covariance
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd GaussNewtonSolverBase::queryCovariance(const steam::StateKey& r,
                                                       const steam::StateKey& c) {

  // Check if the Hessian has been factorized (without augmentation, i.e. the Information matrix)
  if (!factorizedInformationSuccesfully_) {
    // Perform a Cholesky factorization of the approximate Hessian matrix
    this->factorizeHessian();
  }

  // Look up block size of state variables

  // Calculate sparse indices of state variables

  // Use solver to solve for covariance
  Eigen::MatrixXd covariance;

  // Do the backward pass, using the Cholesky factorization (fast)
  //for () {
  //  hessianSolver_.solve(gradientVector_);
  //}

  return covariance;
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

  // Convert to Eigen Type - with the block-sparsity pattern
  // ** Note we do not exploit sub-block-sparsity in case it changes at a later iteration
  approximateHessian_ = A_.toEigen(false);
  gradientVector_ = b_.toEigen();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Solve the Gauss-Newton system of equations: A*x = b
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GaussNewtonSolverBase::solveGaussNewton() {

  // Perform a Cholesky factorization of the approximate Hessian matrix
  this->factorizeHessian();

  // Do the backward pass, using the Cholesky factorization (fast)
  return hessianSolver_.solve(gradientVector_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Solve the Levenberg–Marquardt system of equations:
///        A*x = b, A = (J^T*J + diagonalCoeff*diag(J^T*J))
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GaussNewtonSolverBase::solveGaussNewtonForLM(double diagonalCoeff) {

  // Augment diagonal of the 'hessian' matrix
  // when a newer version of eigen is available, the below line should work:
  //   gaussNewtonLHS_A.diagonal() *= (1.0 + diagonalCoeff);
  for (int i = 0; i < approximateHessian_.outerSize(); i++) {
    approximateHessian_.coeffRef(i,i) *= (1.0 + diagonalCoeff);
  }

  // Solve system
  Eigen::VectorXd levMarqStep;
  try {

    // Solve for the LM step
    levMarqStep = this->solveGaussNewton();

    // Set false because the augmented system is not the information matrix
    factorizedInformationSuccesfully_ = false;

  } catch (const decomp_failure& ex) {

    // Revert diagonal of the 'hessian' matrix
    for (int i = 0; i < approximateHessian_.outerSize(); i++) {
      approximateHessian_.coeffRef(i,i) /= (1.0 + diagonalCoeff);
    }

    // Throw up again
    throw ex;
  }

  // Revert diagonal of the 'hessian' matrix
  // when a newer version of eigen is available, the below line should work:
  //   gaussNewtonLHS_A.diagonal() /= (1.0 + diagonalCoeff);
  for (int i = 0; i < approximateHessian_.outerSize(); i++) {
    approximateHessian_.coeffRef(i,i) /= (1.0 + diagonalCoeff);
  }

  return levMarqStep;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Find the Cauchy point (used for the Dogleg method).
///        The cauchy point is the optimal step length in the gradient descent direction.
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GaussNewtonSolverBase::getCauchyPoint() const {
  double num = gradientVector_.squaredNorm();
  double den = gradientVector_.transpose() *
               (approximateHessian_.selfadjointView<Eigen::Upper>() * gradientVector_);
  return (num/den)*gradientVector_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the predicted cost reduction based on the proposed step
//////////////////////////////////////////////////////////////////////////////////////////////
double GaussNewtonSolverBase::predictedReduction(const Eigen::VectorXd& step) const {
  // b^T * s - 0.5 * s^T * A * s
  double bts = gradientVector_.transpose() * step;
  double stAs = step.transpose() * (approximateHessian_.selfadjointView<Eigen::Upper>() * step);
  return bts - 0.5 * stAs;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Perform the LLT decomposition on the approx. Hessian matrix
//////////////////////////////////////////////////////////////////////////////////////////////
void GaussNewtonSolverBase::factorizeHessian() {

  // Check if the pattern has been initialized
  if (!patternInitialized_) {

    // The first time we are solving the problem we need to analyze the sparsity pattern
    // ** Note we use approximate-minimal-degree (AMD) reordering.
    //    Also, this step does not actually use the numerical values in gaussNewtonLHS
    hessianSolver_.analyzePattern(approximateHessian_);
    patternInitialized_ = true;
  }

  // Perform a Cholesky factorization of the approximate Hessian matrix
  factorizedInformationSuccesfully_ = false;
  hessianSolver_.factorize(approximateHessian_);

  // Check if the factorization succeeded
  if (hessianSolver_.info() != Eigen::Success) {
    throw decomp_failure("During steam solve, Eigen LLT decomposition failed."
                         "It is possible that the matrix was ill-conditioned, in which case "
                         "adding a prior may help. On the other hand, it is also possible that "
                         "the problem you've constructed is not positive semi-definite.");
  } else {
    factorizedInformationSuccesfully_ = true;
  }

  // todo - it would be nice to check the condition number (not just the determinant) of the
  // solved system... need to find a fast way to do this
}

} // steam

