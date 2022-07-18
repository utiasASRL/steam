#pragma once

#include "steam/problem/OptimizationProblem.hpp"
#include "steam/problem/problem.hpp"

namespace steam {

class Covariance {
 public:
  using Ptr = std::shared_ptr<Covariance>;
  using ConstPtr = std::shared_ptr<const Covariance>;

  Covariance(Problem& problem);
  Covariance(const OptimizationProblem& problem);

  virtual ~Covariance() = default;

  Eigen::MatrixXd query(const StateVarBase::ConstPtr& var) const;
  Eigen::MatrixXd query(const StateVarBase::ConstPtr& rvar,
                        const StateVarBase::ConstPtr& cvar) const;
  Eigen::MatrixXd query(const std::vector<StateVarBase::ConstPtr>& vars) const;
  Eigen::MatrixXd query(const std::vector<StateVarBase::ConstPtr>& rvars,
                        const std::vector<StateVarBase::ConstPtr>& cvars) const;

 private:
  StateVector state_vector_;  /// \todo this should be a reference
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>
      hessian_solver_;
};

}  // namespace steam