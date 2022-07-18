//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SimplePoseGraphRelax.cpp
/// \brief A sample usage of the STEAM Engine library for a odometry-style pose
/// graph relaxation
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <thread>

#include <lgmath.hpp>
#include <steam.hpp>

using SE3StateVar = steam::se3::SE3StateVar;

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Structure to store simulated relative transform measurements
//////////////////////////////////////////////////////////////////////////////////////////////
struct RelMeas {
  unsigned int idxA;           // index of pose variable A
  unsigned int idxB;           // index of pose variable B
  SE3StateVar::Ptr meas_T_BA;  // measured transform from A to B
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves a relative pose graph problem
//////////////////////////////////////////////////////////////////////////////////////////////
void runPoseGraphRelax() {
  ///
  /// Setup 'Dataset'
  ///   Here, we simulate a simple odometry-style dataset of relative poses (no
  ///   loop closures). The addition of loop closures is trivial.
  ///

  unsigned int numPoses = 1000;
  std::vector<RelMeas> measCollection;

  // Simulate some measurements
  for (unsigned int i = 1; i < numPoses; i++) {
    // 'Forward' in x with a small angular velocity
    Eigen::Matrix<double, 6, 1> measVec;
    double v_x = -1.0;
    double omega_z = 0.01;
    measVec << v_x, 0.0, 0.0, 0.0, 0.0, omega_z;

    // Create simulated relative measurement
    RelMeas meas;
    meas.idxA = i - 1;
    meas.idxB = i;
    meas.meas_T_BA = SE3StateVar::MakeShared(SE3StateVar::T(measVec));
    meas.meas_T_BA->locked() = true;
    measCollection.push_back(meas);
  }

  // Initialize problem
  steam::OptimizationProblem2 problem;

  ///
  /// Setup States
  ///

  // steam state variables - initialized at identity
  std::vector<SE3StateVar::Ptr> poses;
  for (unsigned int i = 0; i < numPoses; i++) {
    poses.emplace_back(SE3StateVar::MakeShared(SE3StateVar::T()));
    problem.addStateVariable(poses.back());
  }

  ///
  /// Setup Cost Terms
  ///

  // Setup shared noise and loss functions
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  Matrix6d cov = Matrix6d::Identity();
  const auto noise_model = steam::StaticNoiseModel<6>::MakeShared(cov);
  const auto loss_function = steam::L2LossFunc::MakeShared();

  // Lock first pose (otherwise entire solution is 'floating')
  //  **Note: alternatively we could add a prior (UnaryTransformError) to the
  //  first pose.
  poses[0]->locked() = true;

  // Turn measurements into cost terms
  for (unsigned int i = 0; i < measCollection.size(); i++) {
    // Get first referenced state variable
    const auto& stateVarA = poses[measCollection[i].idxA];
    // Get second referenced state variable
    const auto& stateVarB = poses[measCollection[i].idxB];
    // Get transform measurement
    const auto& meas_T_BA = measCollection[i].meas_T_BA;
    // Construct error function
    using namespace steam::se3;
    const auto hat_T_AB = compose(stateVarA, inverse(stateVarB));
    const auto error_function = tran2vec(compose(meas_T_BA, hat_T_AB));

    // Construct cost term
    const auto cost_term = steam::WeightedLeastSqCostTerm<6>::MakeShared(
        error_function, noise_model, loss_function);
    // Add cost term
    problem.addCostTerm(cost_term);
  }

  ///
  /// Setup Solver and Optimize
  ///
  steam::GaussNewtonSolver::Params params;
  params.verbose = true;
  steam::GaussNewtonSolver solver(problem, params);

  // Optimize
  solver.optimize();
}

int main(int argc, char** argv) {
  std::cout << "Test single thread execution." << std::endl;
  runPoseGraphRelax();
#if true
  std::cout << "Test multi thread execution (C++11)." << std::endl;
  std::vector<std::thread> threads;
  for (int i = 1; i <= 10; ++i)
    threads.push_back(std::thread(runPoseGraphRelax));
  for (auto& th : threads) th.join();
#endif
}
