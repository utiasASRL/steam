#pragma once

#include <stdexcept>

#include <Eigen/Core>

#include "steam/problem/problem.hpp"

namespace steam {

/** \brief Reports solver failures (e.g. LLT decomposition fail). */
class solver_failure : public std::runtime_error {
 public:
  solver_failure(const std::string& s) : std::runtime_error(s) {}
};

/** \brief Reports unsuccessful steps (should be indicated to user) */
class unsuccessful_step : public solver_failure {
 public:
  unsuccessful_step(const std::string& s) : solver_failure(s) {}
};

/** \brief Basic solver interface */
class SolverBase {
 public:
  struct Params {
    virtual ~Params() = default;

    /// Whether the solver should be verbose
    bool verbose = false;
    /// Maximum iterations
    unsigned int max_iterations = 100;
    /// Absolute cost threshold to trigger convergence (cost is less than x)
    double absolute_cost_threshold = 0.0;
    /// Change in cost threshold to trigger convergence (cost went down by less
    /// than x)
    double absolute_cost_change_threshold = 1e-4;
    /// Relative cost threshold to trigger convergence (costChange/oldCost is
    /// less than x)
    double relative_cost_change_threshold = 1e-4;
  };

  enum Termination {
    TERMINATE_NOT_YET_TERMINATED,
    TERMINATE_STEP_UNSUCCESSFUL,
    TERMINATE_MAX_ITERATIONS,
    TERMINATE_CONVERGED_ABSOLUTE_ERROR,
    TERMINATE_CONVERGED_ABSOLUTE_CHANGE,
    TERMINATE_CONVERGED_RELATIVE_CHANGE,
    TERMINATE_CONVERGED_ZERO_GRADIENT,
    TERMINATE_COST_INCREASED,
    TERMINATE_EXPECTED_DELTA_COST_CONVERGED,
  };

  SolverBase(Problem& problem, const Params& params);
  virtual ~SolverBase() = default;

  Termination termination_cause() const { return term_; }
  unsigned int curr_iteration() const { return curr_iteration_; }

  /** \brief Perform iterations until convergence */
  void optimize();

 protected:
  /** \brief Perform one iteration */
  void iterate();

  /** \brief Propose an update to the state vector. */
  double proposeUpdate(const Eigen::VectorXd& perturbation);

  /** \brief Confirm the proposed update */
  void acceptProposedState();

  /** \brief Reject the proposed update and revert to the previous values */
  void rejectProposedState();

  StateVector::ConstWeakPtr state_vector() { return state_vector_; }

  /** \brief Reference to optimization problem */
  Problem& problem_;
  /** \brief Collection of state variables */
  const StateVector::WeakPtr state_vector_;
  /** \brief backup state vector for reverting to previous state values */
  StateVector state_vector_backup_;

  Termination term_ = TERMINATE_NOT_YET_TERMINATED;
  unsigned int curr_iteration_ = 0;
  bool solver_converged_ = false;
  double curr_cost_ = 0.0;
  double prev_cost_ = 0.0;
  bool pending_proposed_state_ = false;

 private:
  /** \brief Build the system, solve, and update the state */
  virtual bool linearizeSolveAndUpdate(double& cost, double& grad_norm) = 0;

  const Params params_;
};

/** \brief Print termination cause */
std::ostream& operator<<(std::ostream& out, const SolverBase::Termination& T);

}  // namespace steam
