#pragma once

#include "steam/problem/problem.hpp"

namespace steam {

/** \brief A standard optimization problem */
class OptimizationProblem : public Problem {
 public:
  using Ptr = std::shared_ptr<OptimizationProblem>;
  static Ptr MakeShared(unsigned int num_threads = 1);
  OptimizationProblem(unsigned int num_threads = 1);

  /** \brief Adds a state variable */
  void addStateVariable(const StateVarBase::Ptr& state_var) override;

  /** \brief Add a cost term */
  void addCostTerm(const BaseCostTerm::ConstPtr& cost_term) override;

  /** \brief Get the total number of cost terms */
  unsigned int getNumberOfCostTerms() const override;

  /** \brief Compute the cost from the collection of cost terms */
  double cost() const override;

  /** \brief Get reference to state variables */
  StateVector::Ptr getStateVector() const override;

  /** \brief Fill in the supplied block matrices */
  void buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian,
                             Eigen::VectorXd& gradient_vector) const override;

 private:
  /** \brief Cumber of threads to evaluate cost terms */
  const unsigned int num_threads_;
  /** \brief Collection of cost terms */
  std::vector<BaseCostTerm::ConstPtr> cost_terms_;
  /** \brief Collection of state variables */
  std::vector<StateVarBase::Ptr> state_vars_;

  /** \brief State vector, created when calling get state vector */
  StateVector::Ptr state_vector_ = StateVector::MakeShared();
};

}  // namespace steam