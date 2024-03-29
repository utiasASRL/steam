#pragma once

#include "steam/solver/gauss_newton_solver.hpp"

namespace steam {

class LineSearchGaussNewtonSolver : public GaussNewtonSolver {
 public:
  struct Params : public GaussNewtonSolver::Params {
    /// Amount to decrease step after each backtrack
    double backtrack_multiplier = 0.5;
    /// Maximimum number of times to backtrack before giving up
    unsigned int max_backtrack_steps = 10;
  };

  LineSearchGaussNewtonSolver(Problem& problem, const Params& params);

 private:
  bool linearizeSolveAndUpdate(double& cost, double& grad_norm) override;

  const Params params_;
};

}  // namespace steam
