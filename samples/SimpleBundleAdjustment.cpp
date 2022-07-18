/**
 * \file SimpleBundleAdjustment.cpp
 * \author Sean Anderson, Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 * \brief A sample usage of the STEAM Engine library for a bundle adjustment
 * problem
 */
#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"
#include "steam/data/ParseBA.hpp"

using namespace steam;

int main(int argc, char** argv) {
  ///
  /// Parse Dataset
  ///

  // Get filename
  std::string filename;
  if (argc < 2) {
    filename = "../include/steam/data/stereo_simulated.txt";
    // filename = "../include/steam/data/stereo_simulated_window1.txt";
    // filename = "../include/steam/data/stereo_simulated_window2.txt";
    std::cout << "Parsing default file: " << filename << std::endl << std::endl;
  } else {
    filename = argv[1];
    std::cout << "Parsing file: " << filename << std::endl << std::endl;
  }

  // Load dataset
  // clang-format off
  data::SimpleBaDataset dataset = data::parseSimpleBaDataset(filename);
  std::cout << "Problem has: " << dataset.frames_gt.size() << " poses" << std::endl;
  std::cout << "             " << dataset.land_gt.size() << " landmarks" << std::endl;
  std::cout << "            ~" << double(dataset.meas.size()) / dataset.frames_gt.size()
            << " meas per pose" << std::endl << std::endl;
  // clang-format on

  ///
  /// Make Optimization Problem
  ///

  OptimizationProblem problem;

  ///
  /// Setup and Initialize States
  ///

  // Setup T_cv as a locked se3 state variable
  const auto pose_c_v = se3::SE3StateVar::MakeShared(dataset.T_cv);
  pose_c_v->locked() = true;

  // Ground truth
  std::vector<se3::SE3StateVar::Ptr> poses_gt_k_0;
  std::vector<stereo::HomoPointStateVar::Ptr> landmarks_gt;

  // State variable containers (and related data)
  std::vector<se3::SE3StateVar::Ptr> poses_ic_k_0;
  std::vector<stereo::HomoPointStateVar::Ptr> landmarks_ic;

  // Setup ground-truth poses
  for (unsigned int i = 0; i < dataset.frames_gt.size(); i++)
    poses_gt_k_0.emplace_back(
        se3::SE3StateVar::MakeShared(dataset.frames_gt[i].T_k0));

  // Setup ground-truth landmarks
  for (unsigned int i = 0; i < dataset.land_gt.size(); i++)
    landmarks_gt.emplace_back(
        stereo::HomoPointStateVar::MakeShared(dataset.land_gt[i].point));

  // Setup poses with initial condition
  for (unsigned int i = 0; i < dataset.frames_ic.size(); i++)
    poses_ic_k_0.emplace_back(
        se3::SE3StateVar::MakeShared(dataset.frames_ic[i].T_k0));

  // Lock first pose (otherwise entire solution is 'floating')
  //  **Note: alternatively we could add a prior to the first pose
  poses_ic_k_0[0]->locked() = true;

  // Setup landmarks with initial condition
  for (unsigned int i = 0; i < dataset.land_ic.size(); i++)
    landmarks_ic.emplace_back(
        stereo::HomoPointStateVar::MakeShared(dataset.land_ic[i].point));

  // Add pose variables
  for (unsigned int i = 1; i < poses_ic_k_0.size(); i++)
    problem.addStateVariable(poses_ic_k_0[i]);
  // Add landmark variables
  for (unsigned int i = 0; i < landmarks_ic.size(); i++)
    problem.addStateVariable(landmarks_ic[i]);

  ///
  /// Setup Cost Terms
  ///

  // Setup shared noise and loss function
  const auto sharedCameraNoiseModel =
      StaticNoiseModel<4>::MakeShared(dataset.noise);
  const auto sharedLossFunc = L2LossFunc::MakeShared();

  // Setup camera intrinsics
  const auto sharedIntrinsics = std::make_shared<stereo::CameraIntrinsics>();
  sharedIntrinsics->b = dataset.camParams.b;
  sharedIntrinsics->fu = dataset.camParams.fu;
  sharedIntrinsics->fv = dataset.camParams.fv;
  sharedIntrinsics->cu = dataset.camParams.cu;
  sharedIntrinsics->cv = dataset.camParams.cv;

  // Generate cost terms for camera measurements
  for (unsigned int i = 0; i < dataset.meas.size(); i++) {
    // Get pose reference
    auto& pose_v_0 = poses_ic_k_0[dataset.meas[i].frameID];
    // Get landmark reference
    auto& landmark = landmarks_ic[dataset.meas[i].landID];
    // Construct transform evaluator between landmark frame (inertial) and
    // camera frame
    const auto pose_c_0 = se3::compose(pose_c_v, pose_v_0);

    // Construct error function
    const auto errorfunc = stereo::StereoErrorEvaluator::MakeShared(
        dataset.meas[i].data, sharedIntrinsics, pose_c_0, landmark);

    // Construct cost term
    const auto cost = WeightedLeastSqCostTerm<4>::MakeShared(
        errorfunc, sharedCameraNoiseModel, sharedLossFunc);

    // Add cost term
    problem.addCostTerm(cost);
  }

  ///
  /// Setup Solver and Optimize
  ///
  DoglegGaussNewtonSolver::Params params;
  params.verbose = true;
  DoglegGaussNewtonSolver solver(problem, params);

  // Optimize
  solver.optimize();

  return 0;
}
