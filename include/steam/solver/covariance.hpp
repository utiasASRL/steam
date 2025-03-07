#pragma once

#include "steam/problem/problem.hpp"
#include "steam/solver/gauss_newton_solver.hpp"

namespace steam {

class Covariance {
 public:
  using Ptr = std::shared_ptr<Covariance>;
  using ConstPtr = std::shared_ptr<const Covariance>;

  Covariance(Problem& problem);
  Covariance(GaussNewtonSolver& solver);

  virtual ~Covariance() = default;

  Eigen::MatrixXd query(const StateVarBase::ConstPtr& var) const;
  Eigen::MatrixXd query(const StateVarBase::ConstPtr& rvar,
                        const StateVarBase::ConstPtr& cvar) const;
  Eigen::MatrixXd query(const std::vector<StateVarBase::ConstPtr>& vars) const;
  Eigen::MatrixXd query(const std::vector<StateVarBase::ConstPtr>& rvars,
                        const std::vector<StateVarBase::ConstPtr>& cvars) const;

 private:
  const StateVector::ConstPtr state_vector_;
  using SolverType =
      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper>;
  const std::shared_ptr<SolverType> hessian_solver_;
};

}  // namespace steam