/**
 * \file SimpleBundleAdjustmentRelLand.cpp
 * \author Sean Anderson, Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 * \brief A sample usage of the STEAM Engine library for a bundle adjustment
 * problem with relative landmarks.
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

  // Initialize problem
  OptimizationProblem problem;

  ///
  /// Setup States
  ///

  // Set a fixed identity transform that will be used to initialize landmarks in
  // their parent frame
  const auto tf_identity =
      se3::SE3StateVar::MakeShared(lgmath::se3::Transformation());
  tf_identity->locked() = true;

  // Fixed vehicle to camera transform
  const auto tf_c_v = se3::SE3StateVar::MakeShared(dataset.T_cv);
  tf_c_v->locked() = true;

  // Ground truth
  std::vector<se3::SE3StateVar::Ptr> poses_gt_k_0;
  std::vector<stereo::HomoPointStateVar::Ptr> landmarks_gt;

  // State variable containers (and related data)
  std::vector<se3::SE3StateVar::Ptr> poses_ic_k_0;
  std::vector<stereo::HomoPointStateVar::Ptr> landmarks_ic;

  // Record the frame in which the landmark is first seen in order to set up
  // transforms correctly
  std::map<unsigned, unsigned> landmark_map;

  ///
  /// Initialize States
  ///

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
  //  **Note: alternatively we could add a prior to the first pose.
  poses_ic_k_0[0]->locked() = true;

  // Setup relative landmarks
  landmarks_ic.resize(dataset.land_ic.size(), nullptr);
  for (unsigned int i = 0; i < dataset.meas.size(); i++) {
    // Get pose reference
    unsigned int frameIdx = dataset.meas[i].frameID;

    // Get landmark reference
    unsigned int landmarkIdx = dataset.meas[i].landID;

    // Setup landmark if first time
    if (!landmarks_ic[landmarkIdx]) {
      // Get homogeneous point in inertial frame
      Eigen::Vector4d p_0;
      p_0.head<3>() = dataset.land_ic[landmarkIdx].point;
      p_0[3] = 1.0;

      // Get transform between first observation time and inertial frame
      lgmath::se3::Transformation pose_vk_0 =
          dataset.frames_ic[frameIdx].T_k0 / dataset.frames_ic[0].T_k0;

      // Get point in 'local' frame
      Eigen::Vector4d p_vehicle = pose_vk_0 * p_0;

      // Insert the landmark
      landmarks_ic[landmarkIdx] =
          stereo::HomoPointStateVar::MakeShared(p_vehicle.head<3>());

      // Keep a record of its 'parent' frame
      landmark_map[landmarkIdx] = frameIdx;
    }
  }

  // Add pose variables
  for (unsigned int i = 0; i < poses_ic_k_0.size(); i++)
    problem.addStateVariable(poses_ic_k_0[i]);
  // Add landmark variables
  for (unsigned int i = 0; i < landmarks_ic.size(); i++)
    problem.addStateVariable(landmarks_ic[i]);

  ///
  /// Setup Cost Terms
  ///

  // Setup shared noise and loss function
  const auto sharedCameraNoiseModel =
      std::make_shared<StaticNoiseModel<4>>(dataset.noise);
  const auto sharedLossFunc = std::make_shared<L2LossFunc>();

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
    unsigned int frameIdx = dataset.meas[i].frameID;
    // Get landmark reference
    unsigned int landmarkIdx = dataset.meas[i].landID;
    auto& landmark = landmarks_ic[landmarkIdx];

    // Construct transform evaluator between two vehicle frames (a and b) that
    // have observations
    Evaluable<lgmath::se3::Transformation>::Ptr tf_vb_va;
    if (landmark_map[landmarkIdx] == frameIdx) {
      // In this case, the transform remains fixed as an identity transform
      tf_vb_va = tf_identity;
    } else {
      unsigned int firstObsIndex = landmark_map[landmarkIdx];
      tf_vb_va = se3::compose(poses_ic_k_0[frameIdx],
                              se3::inverse(poses_ic_k_0[firstObsIndex]));
    }

    // Compose with camera to vehicle transform
    const auto tf_cb_va = se3::compose(tf_c_v, tf_vb_va);

    // Construct error function
    const auto errorfunc = stereo::StereoErrorEvaluator::MakeShared(
        dataset.meas[i].data, sharedIntrinsics, tf_cb_va, landmark);

    // Construct cost term
    const auto cost = std::make_shared<WeightedLeastSqCostTerm<4>>(
        errorfunc, sharedCameraNoiseModel, sharedLossFunc);

    // Add cost term
    problem.addCostTerm(cost);
  }

  ///
  /// Setup Solver and Optimize
  ///
  using SolverType = DoglegGaussNewtonSolver;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  return 0;
}