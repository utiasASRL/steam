//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SimplePointCloudAlignment.cpp
/// \brief
///
/// \author Yuchen Wu, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>

int main(int argc, char **argv) {
  using namespace steam;

  /// Reference points to align against
  Eigen::Matrix<double, 4, 9> ref_pts;
  // clang-format off
  ref_pts << 0,   1,   1,  0, -1, -1, -1,  0,  1,
             0,   0,   1,  1,  1,  0, -1, -1, -1,
           0.1, 0.2, 0.3,  0,  0,  0,  0,  0,  0,
             1,   1,   1,  1,  1,  1,  1,  1,  1;
  // clang-format on

  /// Ground truth pose
  Eigen::Matrix<double, 6, 1> T_mq_vec;
  T_mq_vec << 1, 1, 1, 1, 1, 1;
  lgmath::se3::Transformation T_mq(T_mq_vec);

  Eigen::Matrix<double, 4, 9> qry_pts = T_mq.inverse().matrix() * ref_pts;

  // Initialize problem
  OptimizationProblem2 problem;

  // state and evaluator
  using SE3StateVar = se3::SE3StateVar;
  const auto T_mq_var = SE3StateVar::MakeShared(SE3StateVar::T());
  problem.addStateVariable(T_mq_var);

  // shared noise and loss functions
  Eigen::Matrix3d cov = Eigen::Matrix3d::Identity();
  const auto noise_model = StaticNoiseModel<3>::MakeShared(cov);
  const auto loss_function = L2LossFunc::MakeShared();

  // cost terms
  for (int i = 0; i < ref_pts.cols(); i++) {
    // Construct error function
    const auto error_function = p2p::p2pError(
        T_mq_var, ref_pts.block<3, 1>(0, i), qry_pts.block<3, 1>(0, i));
    // Construct cost term
    const auto cost_term = WeightedLeastSqCostTerm<3>::MakeShared(
        error_function, noise_model, loss_function);
    // Add cost term
    problem.addCostTerm(cost_term);
  }

  // Make solver
  GaussNewtonSolver::Params params;
  params.verbose = true;
  GaussNewtonSolver solver(problem, params);

  // Optimize
  solver.optimize();

  std::cout << "true T_mq:\n" << T_mq << std::endl;
  std::cout << "estimated T_mq:\n" << T_mq_var->value() << std::endl;

  return 0;
}