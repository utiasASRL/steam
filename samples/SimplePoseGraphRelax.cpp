//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SimplePoseGraphRelax.cpp
/// \brief A sample usage of the STEAM Engine library for a odometry-style pose graph relaxation
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Structure to store simulated relative transform measurements
//////////////////////////////////////////////////////////////////////////////////////////////
struct RelMeas {
  unsigned int idxA; // index of pose variable A
  unsigned int idxB; // index of pose variable B
  lgmath::se3::Transformation meas_T_BA; // measured transform from A to B
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves a relative pose graph problem
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  ///
  /// Setup 'Dataset'
  ///   Here, we simulate a simple odometry-style dataset of relative poses (no loop closures).
  ///   The addition of loop closures is trivial.
  ///

  unsigned int numPoses = 1000;
  std::vector<RelMeas> measCollection;

  // Simulate some measurements
  for (unsigned int i = 1; i < numPoses; i++)
  {
    // 'Forward' in x with a small angular velocity
    Eigen::Matrix<double,6,1> measVec;
    double v_x = -1.0;
    double omega_z = 0.01;
    measVec << v_x, 0.0, 0.0, 0.0, 0.0, omega_z;

    // Create simulated relative measurement
    RelMeas meas;
    meas.idxA = i-1;
    meas.idxB = i;
    meas.meas_T_BA = lgmath::se3::Transformation(measVec);
    measCollection.push_back(meas);
  }

  ///
  /// Setup States
  ///

  // steam state variables
  std::vector<steam::se3::TransformStateVar::Ptr> poses;

  // Setup state variables - initialized at identity
  for (unsigned int i = 0; i < numPoses; i++)
  {
    steam::se3::TransformStateVar::Ptr temp(new steam::se3::TransformStateVar());
    poses.push_back(temp);
  }

  ///
  /// Setup Cost Terms
  ///

  // steam cost terms
  std::vector<steam::CostTermX::Ptr> costTerms;

  // Setup shared noise and loss functions
  steam::NoiseModelX::Ptr sharedNoiseModel(new steam::NoiseModelX(Eigen::MatrixXd::Identity(6,6)));
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Lock first pose (otherwise entire solution is 'floating')
  //  **Note: alternatively we could add a prior (UnaryTransformError) to the first pose.
  poses[0]->setLock(true);

  // Turn measurements into cost terms
  for (unsigned int i = 0; i < measCollection.size(); i++)
  {
    // Get first referenced state variable
    steam::se3::TransformStateVar::Ptr& stateVarA = poses[measCollection[i].idxA];

    // Get second referenced state variable
    steam::se3::TransformStateVar::Ptr& stateVarB = poses[measCollection[i].idxB];

    // Get transform measurement
    lgmath::se3::Transformation& meas_T_BA = measCollection[i].meas_T_BA;

    // Construct error function
    steam::TransformErrorEval::Ptr errorfunc(new steam::TransformErrorEval(meas_T_BA, stateVarB, stateVarA));

    // Create cost term and add to problem
    steam::CostTermX::Ptr cost(new steam::CostTermX(errorfunc, sharedNoiseModel, sharedLossFunc));
    costTerms.push_back(cost);
  }

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 1; i < poses.size(); i++)
  {
    problem.addStateVariable(poses[i]);
  }

  // Add cost terms
  for (unsigned int i = 0; i < costTerms.size(); i++)
  {
    problem.addCostTerm(costTerms[i]);
  }

  ///
  /// Setup Solver and Optimize
  ///
  typedef steam::VanillaGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  return 0;
}

