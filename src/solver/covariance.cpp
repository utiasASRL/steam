#include "steam/solver/covariance.hpp"

#include "steam/blockmat/BlockMatrix.hpp"
#include "steam/blockmat/BlockSparseMatrix.hpp"
#include "steam/blockmat/BlockVector.hpp"

namespace steam {

Covariance::Covariance(const OptimizationProblem& problem) {
  const auto& vars = problem.getStateVariables();
  for (const auto& var : vars) {
    if (!var->locked()) state_vec_.addStateVariable(var);
  }

  Eigen::SparseMatrix<double> approx_hessian;
  Eigen::VectorXd gradient_vector;
  problem.buildGaussNewtonTerms(state_vec_, &approx_hessian, &gradient_vector);

  hessian_solver_.analyzePattern(approx_hessian);
  hessian_solver_.factorize(approx_hessian);
  if (hessian_solver_.info() != Eigen::Success) {
    throw std::runtime_error(
        "During steam solve, Eigen LLT decomposition failed. "
        "It is possible that the matrix was ill-conditioned, in which case "
        "adding a prior may help. On the other hand, it is also possible that "
        "the problem you've constructed is not positive semi-definite.");
  }
}

Eigen::MatrixXd Covariance::query(const StateVarBase::ConstPtr& var) const {
  return query(std::vector<StateVarBase::ConstPtr>{var});
}

Eigen::MatrixXd Covariance::query(const StateVarBase::ConstPtr& rvar,
                                  const StateVarBase::ConstPtr& cvar) const {
  return query(std::vector<StateVarBase::ConstPtr>{rvar},
               std::vector<StateVarBase::ConstPtr>{cvar});
}

Eigen::MatrixXd Covariance::query(
    const std::vector<StateVarBase::ConstPtr>& vars) const {
  return query(vars, vars);
}

Eigen::MatrixXd Covariance::query(
    const std::vector<StateVarBase::ConstPtr>& rvars,
    const std::vector<StateVarBase::ConstPtr>& cvars) const {
  // Creating indexing
  BlockMatrixIndexing indexing(state_vec_.getStateBlockSizes());
  const auto& blk_row_indexing = indexing.rowIndexing();
  const auto& blk_col_indexing = indexing.colIndexing();

  // Fixed sizes
  const auto num_row_vars = rvars.size();
  const auto num_col_vars = cvars.size();

  // Look up block indexes
  std::vector<unsigned int> blk_row_indices;
  blk_row_indices.reserve(num_row_vars);
  for (size_t i = 0; i < num_row_vars; i++)
    blk_row_indices.emplace_back(
        state_vec_.getStateBlockIndex(rvars[i]->key()));

  std::vector<unsigned int> blk_col_indices;
  blk_col_indices.reserve(num_col_vars);
  for (size_t i = 0; i < num_col_vars; i++)
    blk_col_indices.emplace_back(
        state_vec_.getStateBlockIndex(cvars[i]->key()));

  // Look up block size of state variables
  std::vector<unsigned int> blk_row_sizes;
  blk_row_sizes.reserve(num_row_vars);
  for (size_t i = 0; i < num_row_vars; i++)
    blk_row_sizes.emplace_back(blk_row_indexing.blkSizeAt(blk_row_indices[i]));

  std::vector<unsigned int> blk_col_sizes;
  blk_col_sizes.resize(num_col_vars);
  for (size_t i = 0; i < num_col_vars; i++)
    blk_col_sizes.emplace_back(blk_col_indexing.blkSizeAt(blk_col_indices[i]));

  // Create result container
  BlockMatrix blk_cov(blk_row_sizes, blk_col_sizes);

  // For each column key
  for (unsigned int c = 0; c < num_col_vars; c++) {
    // For each scalar column
    Eigen::VectorXd projection(blk_row_indexing.scalarSize());
    projection.setZero();
    for (unsigned int j = 0; j < blk_col_sizes[c]; j++) {
      // Get scalar index
      unsigned int scalar_col_index =
          blk_col_indexing.cumSumAt(blk_col_indices[c]) + j;

      // Solve for scalar column of covariance matrix
      projection(scalar_col_index) = 1.0;
      Eigen::VectorXd x = hessian_solver_.solve(projection);
      projection(scalar_col_index) = 0.0;

      // For each block row
      for (unsigned int r = 0; r < num_row_vars; r++) {
        // Get scalar index into solution vector
        int scalarRowIndex = blk_row_indexing.cumSumAt(blk_row_indices[r]);

        // Do the backward pass, using the Cholesky factorization (fast)
        blk_cov.at(r, c).block(0, j, blk_row_sizes[r], 1) =
            x.block(scalarRowIndex, 0, blk_row_sizes[r], 1);
      }
    }
  }

  // To Eigen format
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(blk_row_indexing.scalarSize(),
                                              blk_col_indexing.scalarSize());
  for (unsigned int r = 0; r < num_row_vars; r++) {
    for (unsigned int c = 0; c < num_col_vars; c++) {
      cov.block(blk_row_indexing.cumSumAt(blk_row_indices[r]),
                blk_col_indexing.cumSumAt(blk_col_indices[c]), blk_row_sizes[r],
                blk_col_sizes[c]) = blk_cov.at(r, c);
    }
  }

  return cov;
}

}  // namespace steam