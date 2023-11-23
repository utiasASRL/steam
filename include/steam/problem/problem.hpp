#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/state_vector.hpp"

namespace steam {

/** \brief Problem interface required by solver */
class Problem {
 public:
  virtual ~Problem() = default;
  using Ptr = std::shared_ptr<Problem>;

  /** \brief Get the total number of cost terms */
  virtual unsigned int getNumberOfCostTerms() const = 0;

  virtual void addStateVariable(const StateVarBase::Ptr& state_var) = 0;

  virtual void addCostTerm(const BaseCostTerm::ConstPtr& cost_term) = 0;

  /** \brief Compute the cost from the collection of cost terms */
  virtual double cost() const = 0;

  /** \brief Get reference to state variables */
  virtual StateVector::Ptr getStateVector() const = 0;

  /** \brief Fill in the supplied block matrices */
  virtual void buildGaussNewtonTerms(
      Eigen::SparseMatrix<double>& approximate_hessian,
      Eigen::VectorXd& gradient_vector) const = 0;
};

}  // namespace steam