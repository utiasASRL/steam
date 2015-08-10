//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SimpleBundleAdjustmentRelLand.cpp
/// \brief A sample usage of the STEAM Engine library for a bundle adjustment problem
///        with relative landmarks.
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <steam/data/ParseBA.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves simple bundle adjustment problems
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  ///
  /// Parse Dataset - sphere of relative pose measurements (fairly dense loop closures)
  ///

  // Get filename
  std::string filename;
  if (argc < 2) {
    filename = "../../include/steam/data/stereo_simulated.txt";
    //filename = "../../include/steam/data/stereo_simulated_window1.txt";
    //filename = "../../include/steam/data/stereo_simulated_window2.txt";
    std::cout << "Parsing default file: " << filename << std::endl << std::endl;
  } else {
    filename = argv[1];
    std::cout << "Parsing file: " << filename << std::endl << std::endl;
  }

  // Load dataset
  steam::data::SimpleBaDataset dataset = steam::data::parseSimpleBaDataset(filename);
  std::cout << "Problem has: " << dataset.frames_gt.size() << " poses" << std::endl;
  std::cout << "             " << dataset.land_gt.size() << " landmarks" << std::endl;
  std::cout << "            ~" << double(dataset.meas.size())/dataset.frames_gt.size() << " meas per pose" << std::endl << std::endl;

  ///
  /// Setup and Initialize States
  ///

  // Ground truth
  std::vector<steam::se3::TransformStateVar::Ptr> poses_gt_k_0;
  std::vector<steam::se3::LandmarkStateVar::Ptr> landmarks_gt;

  // State variable containers (and related data)
  std::vector<steam::se3::TransformStateVar::Ptr> poses_ic_k_0;
  std::vector<steam::se3::LandmarkStateVar::Ptr> landmarks_ic;

  // Setup ground-truth poses
  for (unsigned int i = 0; i < dataset.frames_gt.size(); i++) {
    steam::se3::TransformStateVar::Ptr temp(new steam::se3::TransformStateVar(dataset.frames_gt[i].T_k0));
    poses_gt_k_0.push_back(temp);
  }

  // Setup ground-truth landmarks
  for (unsigned int i = 0; i < dataset.land_gt.size(); i++) {
    steam::se3::LandmarkStateVar::Ptr temp(new steam::se3::LandmarkStateVar(dataset.land_gt[i].point));
    landmarks_gt.push_back(temp);
  }

  // Setup poses with initial condition
  for (unsigned int i = 0; i < dataset.frames_ic.size(); i++) {
    steam::se3::TransformStateVar::Ptr temp(new steam::se3::TransformStateVar(dataset.frames_ic[i].T_k0));
    poses_ic_k_0.push_back(temp);
  }

  // Lock first pose (otherwise entire solution is 'floating')
  //  **Note: alternatively we could add a prior (UnaryTransformError) to the first pose.
  poses_ic_k_0[0]->setLock(true);

  ///
  /// Setup Cost Terms
  ///

  // steam cost terms
  std::vector<steam::CostTermX::Ptr> costTerms;

  // Setup shared noise and loss function
  steam::NoiseModelX::Ptr sharedCameraNoiseModel(new steam::NoiseModelX(dataset.noise));
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Setup camera intrinsics
  steam::StereoCameraErrorEvalX::CameraIntrinsics::Ptr sharedIntrinsics(
        new steam::StereoCameraErrorEvalX::CameraIntrinsics());
  sharedIntrinsics->b  = dataset.camParams.b;
  sharedIntrinsics->fu = dataset.camParams.fu;
  sharedIntrinsics->fv = dataset.camParams.fv;
  sharedIntrinsics->cu = dataset.camParams.cu;
  sharedIntrinsics->cv = dataset.camParams.cv;

  // Size vector for landmarks -- initialize while going through measurements
  landmarks_ic.resize(dataset.land_ic.size());

  // Generate cost terms for camera measurements
  for (unsigned int i = 0; i < dataset.meas.size(); i++) {

    // Get pose reference
    unsigned int frameIdx = dataset.meas[i].frameID;
    steam::se3::TransformStateVar::Ptr& poseVar = poses_ic_k_0[frameIdx];

    // Setup landmark if first time
    unsigned int landmarkIdx = dataset.meas[i].landID;
    if (!landmarks_ic[landmarkIdx]) {
      Eigen::Vector4d p_v0; p_v0.head<3>() = dataset.land_ic[landmarkIdx].point; p_v0[3] = 1.0;
      Eigen::Vector4d p_vl = (poses_ic_k_0[frameIdx]->getValue()/poses_ic_k_0[0]->getValue()) * p_v0;
      landmarks_ic[landmarkIdx] = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p_vl.head<3>(), steam::se3::TransformStateEvaluator::MakeShared(poses_ic_k_0[frameIdx])));
    }

    // Get landmark reference
    steam::se3::LandmarkStateVar::Ptr& landVar = landmarks_ic[landmarkIdx];

    // Construct transform evaluator between landmark frame (inertial) and camera frame
    steam::se3::TransformEvaluator::Ptr pose_c_v = steam::se3::FixedTransformEvaluator::MakeShared(dataset.T_cv);
    steam::se3::TransformEvaluator::Ptr pose_vk_0 = steam::se3::TransformStateEvaluator::MakeShared(poseVar);
    steam::se3::TransformEvaluator::Ptr pose_c_0 = steam::se3::compose(pose_c_v, pose_vk_0);

    // Construct error function
    steam::StereoCameraErrorEvalX::Ptr errorfunc(new steam::StereoCameraErrorEvalX(
            dataset.meas[i].data, sharedIntrinsics, pose_c_0, landVar));

    // Construct cost term
    steam::CostTermX::Ptr cost(new steam::CostTermX(errorfunc, sharedCameraNoiseModel, sharedLossFunc));
    costTerms.push_back(cost);
  }

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add pose variables
  for (unsigned int i = 1; i < poses_ic_k_0.size(); i++) {
    problem.addStateVariable(poses_ic_k_0[i]);
  }

  // Add landmark variables
  for (unsigned int i = 0; i < landmarks_ic.size(); i++) {
    problem.addStateVariable(landmarks_ic[i]);
  }

  // Add cost terms
  for (unsigned int i = 0; i < costTerms.size(); i++) {
    problem.addCostTerm(costTerms[i]);
  }

  ///
  /// Setup Solver and Optimize
  ///
  typedef steam::DoglegGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  return 0;
}
