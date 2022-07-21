#pragma once

#include <deque>

#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/problem.hpp"

namespace steam {

class SlidingWindowFilter : public Problem {
  struct Variable {
    Variable(const StateVarBase::Ptr& v, bool m)
        : variable(v), marginalize(m) {}
    StateVarBase::Ptr variable = nullptr;
    bool marginalize = false;
  };
  using VariableMap = std::unordered_map<StateKey, Variable, StateKeyHash>;
  using KeySet = BaseCostTerm::KeySet;

 public:
  SlidingWindowFilter(unsigned int num_threads = 1);

  // for debugging
  const VariableMap& variables() const { return variables_; }

  void addVariable(const StateVarBase::Ptr& variable);
  void addVariable(const std::vector<StateVarBase::Ptr>& variables);

  void marginalizeVariable(const StateVarBase::Ptr& variable);
  void marginalizeVariable(const std::vector<StateVarBase::Ptr>& variables);

  void addCostTerm(const BaseCostTerm::ConstPtr& cost_term);

  /** \brief Compute the cost from the collection of cost terms */
  double cost() const override;

  /** \brief Get the total number of cost terms */
  unsigned int getNumberOfCostTerms() const override;

  /** \brief Get reference to state variables */
  StateVector::Ptr getStateVector() override;

  /** \brief Fill in the supplied block matrices */
  void buildGaussNewtonTerms(Eigen::SparseMatrix<double>& approximate_hessian,
                             Eigen::VectorXd& gradient_vector) override;

 private:
  /** \brief Cumber of threads to evaluate cost terms */
  const unsigned int num_threads_;

  VariableMap variables_;
  std::deque<StateKey> variable_queue_;
  /** \brief Keeps track of */
  std::unordered_map<StateKey, KeySet, StateKeyHash> related_var_keys_;
  /** \brief Collection of cost terms */
  std::vector<BaseCostTerm::ConstPtr> cost_terms_;

  /// \todo store as sparse matrix
  Eigen::MatrixXd fixed_A_;
  Eigen::VectorXd fixed_b_;

  //
  const StateVector::Ptr marginalize_state_vector_ = StateVector::MakeShared();
  const StateVector::Ptr active_state_vector_ = StateVector::MakeShared();
  const StateVector::Ptr state_vector_ = StateVector::MakeShared();
};

}  // namespace steam