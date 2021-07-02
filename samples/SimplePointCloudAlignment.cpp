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

  // state and evaluator
  steam::se3::TransformStateVar::Ptr T_mq_var(
      new steam::se3::TransformStateVar());
  steam::se3::TransformStateEvaluator::ConstPtr T_mq_eval =
      steam::se3::TransformStateEvaluator::MakeShared(T_mq_var);

  // shared noise and loss functions
  steam::BaseNoiseModel<3>::Ptr noise_model(
      new steam::StaticNoiseModel<3>(Eigen::MatrixXd::Identity(3, 3)));
  steam::L2LossFunc::Ptr loss_func(new steam::L2LossFunc());

  // cost terms
  steam::ParallelizedCostTermCollection::Ptr cost_terms(
      new steam::ParallelizedCostTermCollection());
  for (int i = 0; i < ref_pts.cols(); i++) {
    // Construct error function
    steam::PointToPointErrorEval2::Ptr error_func(
        new steam::PointToPointErrorEval2(T_mq_eval, ref_pts.block<3, 1>(0, i),
                                          qry_pts.block<3, 1>(0, i)));

    // Create cost term and add to problem
    steam::WeightedLeastSqCostTerm<3, 6>::Ptr cost(
        new steam::WeightedLeastSqCostTerm<3, 6>(error_func, noise_model,
                                                 loss_func));
    cost_terms->add(cost);
  }

  // Initialize problem
  steam::OptimizationProblem problem;
  problem.addStateVariable(T_mq_var);
  problem.addCostTerm(cost_terms);

  typedef steam::VanillaGaussNewtonSolver SolverType;
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  std::cout << "true T_mq:\n" << T_mq << std::endl;
  std::cout << "estimated T_mq:\n" << T_mq_var->getValue() << std::endl;

  return 0;
}