/**
 * \file BSplineTrajectoryExample.cpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#include <iomanip>
#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace steam;

/** \brief Example that loads and solves simple bundle adjustment problems */
int main(int argc, char** argv) {
  const double T = 1.0;
  traj::Time knot_spacing(0.4);

  // Create a trajectory interface
  traj::bspline::Interface traj(knot_spacing);

  // clang-format off
  std::vector<std::pair<traj::Time, Eigen::Matrix<double, 6, 1>>> w_iv_inv_meas;
  w_iv_inv_meas.emplace_back(traj::Time(0.1 * T), 0.0 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.2 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.3 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.4 * T), 0.6 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.5 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.6 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.7 * T), 0.0 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.8 * T), 0.2 * Eigen::Matrix<double, 6, 1>::Ones());
  w_iv_inv_meas.emplace_back(traj::Time(0.9 * T), 0.4 * Eigen::Matrix<double, 6, 1>::Ones());

  std::vector<BaseCostTerm::Ptr> cost_terms;
  const auto loss_func = L2LossFunc::MakeShared();
  const auto noise_model = StaticNoiseModel<6>::MakeShared(Eigen::Matrix<double, 6, 6>::Identity());
  for (auto& meas : w_iv_inv_meas) {
    const auto error_func = vspace::vspace_error<6>(traj.getVelocityInterpolator(meas.first), meas.second);
    cost_terms.emplace_back(WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func));
  }
  // clang-format on

  //
  OptimizationProblem problem(1);
  // add state variables
  traj.addStateVariables(problem);
  // add trajectory cost terms
  traj.addPriorCostTerms(problem);
  // add meas cost terms
  for (const auto& cost : cost_terms) problem.addCostTerm(cost);

  using SolverType = VanillaGaussNewtonSolver;
  SolverType::Params params;
  params.verbose = true;
  SolverType solver(&problem, params);
  solver.optimize();

  return 0;
}