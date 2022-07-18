#pragma once

#include <stdexcept>

#include <Eigen/Core>

#include "steam/problem/problem.hpp"

namespace steam {

/** \brief Reports solver failures (e.g. LLT decomposition fail). */
class solver_failure2 : public std::runtime_error {
 public:
  solver_failure2(const std::string& s) : std::runtime_error(s) {}
};

/** \brief Reports unsuccessful steps (should be indicated to user) */
class unsuccessful_step2 : public solver_failure2 {
 public:
  unsuccessful_step2(const std::string& s) : solver_failure2(s) {}
};

/** \brief Basic solver interface */
class SolverBase2 {
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
    TERMINATE_CONVERGED_ZERO_GRADIENT
  };

  SolverBase2(Problem& problem, const Params& params);
  virtual ~SolverBase2() = default;

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

  /** \brief Reference to optimization problem */
  Problem& problem_;
  /** \brief Collection of state variables */
  StateVector& state_vector_;
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
std::ostream& operator<<(std::ostream& out, const SolverBase2::Termination& T);

}  // namespace steam
