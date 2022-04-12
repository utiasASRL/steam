//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SpherePoseGraphRelax.cpp
/// \brief A sample usage of the STEAM Engine library for solving the iSAM1
/// spherical pose graph relaxation problem.
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <steam/data/ParseSphere.hpp>

using SE3StateVar = steam::se3::SE3StateVar;

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves an iSAM1 spherical pose graph problem
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  ///
  /// Parse Dataset - sphere of relative pose measurements (fairly dense loop
  /// closures)
  ///

  // Get filename
  std::string filename;
  if (argc < 2) {
    filename = "../include/steam/data/sphere2500.txt";
    std::cout << "Parsing default file: " << filename << std::endl << std::endl;
  } else {
    filename = argv[1];
    std::cout << "Parsing file: " << filename << std::endl << std::endl;
  }

  std::vector<steam::data::SphereEdge> measCollection =
      steam::data::parseSphereDataset(filename);

  // Initialize problem
  steam::OptimizationProblem problem;

  ///
  /// Setup and Initialize States
  ///

  // steam state variables
  std::vector<SE3StateVar::Ptr> poses_k_0;

  // Edges (for graphics)
  std::vector<std::pair<int, int>> edges;

  // Add initial state
  {
    const auto pose_0_0 = SE3StateVar::MakeShared(SE3StateVar::T());

    // Lock first pose (otherwise entire solution is 'floating')
    //  **Note: alternatively we could add a prior (UnaryTransformError) to the
    //  first pose.
    pose_0_0->locked() = true;

    poses_k_0.push_back(pose_0_0);
    problem.addStateVariable(poses_k_0.back());
  }

  // Add states from odometry
  for (unsigned int i = 0; i < measCollection.size(); i++) {
    // Looping through all measurements (including loop closures), check if
    // measurement provides odometry
    if (measCollection[i].idA == poses_k_0.size() - 1 &&
        measCollection[i].idB == poses_k_0.size()) {
      lgmath::se3::Transformation T_k_0 =
          measCollection[i].T_BA * poses_k_0[poses_k_0.size() - 1]->value();
      const auto temp = SE3StateVar::MakeShared(T_k_0);
      poses_k_0.push_back(temp);
      problem.addStateVariable(poses_k_0.back());
    }

    // Add edge graphic
    edges.push_back(
        std::make_pair(measCollection[i].idA, measCollection[i].idB));
  }

  ///
  /// Setup Cost Terms
  ///

  // Setup shared noise and loss functions
  const auto sharedNoiseModel = steam::StaticNoiseModel<6>::MakeShared(
      measCollection[0].sqrtInformation, steam::NoiseType::SQRT_INFORMATION);
  const auto sharedLossFunc = steam::L2LossFunc::MakeShared();

  // Turn measurements into cost terms
  for (unsigned int i = 0; i < measCollection.size(); i++) {
    // Get first referenced state variable
    const auto& stateVarA = poses_k_0[measCollection[i].idA];
    // Get second referenced state variable
    const auto& stateVarB = poses_k_0[measCollection[i].idB];
    // Get transform measurement
    const auto meas_T_BA = SE3StateVar::MakeShared(measCollection[i].T_BA);
    meas_T_BA->locked() = true;
    // Construct error function
    using namespace steam::se3;
    const auto hat_T_AB = compose(stateVarA, inverse(stateVarB));
    const auto error_function = tran2vec(compose(meas_T_BA, hat_T_AB));

    // Construct cost term
    const auto cost_term = steam::WeightedLeastSqCostTerm<6>::MakeShared(
        error_function, sharedNoiseModel, sharedLossFunc);
    // Add cost term
    problem.addCostTerm(cost_term);
  }

  ///
  /// Setup Solver and Optimize
  ///
  using SolverType = steam::DoglegGaussNewtonSolver;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  return 0;
}
