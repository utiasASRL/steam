#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "steam/evaluable/state_var.hpp"
#include "steam/problem/CostTermBase.hpp"
#include "steam/problem/StateVec.hpp"

namespace steam {

////////////////////////////////////////////////////////////////////////////////
/// The define STEAM_DEFAULT_NUM_OPENMP_THREADS can be used to set the default
/// template parameter for the number of threads that process a collection of
/// cost terms. Note that this define can be set in CMake with the following
/// command:
///
/// add_definitions(-DSTEAM_DEFAULT_NUM_OPENMP_THREADS=4)
///
/// If it is not user defined, we default it to 4.
////////////////////////////////////////////////////////////////////////////////
#ifndef STEAM_DEFAULT_NUM_OPENMP_THREADS
#define STEAM_DEFAULT_NUM_OPENMP_THREADS 4
#endif

/**
 * \brief Container for active state variables and cost terms associated with
 * the optimization problem to be solved.
 */
class OptimizationProblem {
 public:
  OptimizationProblem(
      unsigned int num_threads = STEAM_DEFAULT_NUM_OPENMP_THREADS);

  /** \brief Adds a state variable to the internal container */
  void addStateVariable(const StateVarBase::Ptr& statevar);

  /** \brief Get reference to state variables */
  const std::vector<StateVarBase::Ptr>& getStateVariables() const;

  /**
   * \brief Add a cost term (should depend on active states that were added to
   * the problem)
   */
  void addCostTerm(const CostTermBase::ConstPtr& costTerm);

  /** \brief Get the total number of cost terms */
  unsigned int getNumberOfCostTerms() const;

  /** \brief Compute the cost from the collection of cost terms */
  double cost() const;

  /** \brief Fill in the supplied block matrices */
  void buildGaussNewtonTerms(const StateVec& state_vector,
                             Eigen::SparseMatrix<double>* approximate_hessian,
                             Eigen::VectorXd* gradient_vector) const;

 private:
  /** \brief Cumber of threads to evaluate cost terms */
  const unsigned int num_threads_;

  /** \brief Collection of cost terms */
  std::vector<CostTermBase::ConstPtr> cost_terms_;

  /** \brief Collection of state variables */
  std::vector<StateVarBase::Ptr> state_vars_;
};

}  // namespace steam
