//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SimpleBundleAdjustmentFullRel.cpp
/// \brief A sample usage of the STEAM Engine library for a bundle adjustment problem
///        with relative landmarks and poses.
///
/// \author Michael Warren, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <glog/logging.h>

#include <lgmath.hpp>

#include <lgmath.hpp>
#include <steam.hpp>
#include <steam/data/ParseBA.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves simple bundle adjustment problems
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  // Init glog
  google::InitGoogleLogging(argv[0]);

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

  // fixed vehicle to camera transform
  steam::se3::TransformEvaluator::Ptr pose_c_v = steam::se3::FixedTransformEvaluator::MakeShared(dataset.T_cv);

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

  // Setup globally consistent poses with initial condition
  for (unsigned int i = 0; i < dataset.frames_ic.size(); i++) {
    steam::se3::TransformStateVar::Ptr temp(new steam::se3::TransformStateVar(dataset.frames_ic[i].T_k0));
    poses_ic_k_0.push_back(temp);
  }

  // Make a relative set of transform state variables
  std::vector<steam::se3::TransformStateVar::Ptr> relposes_ic_k_kp;

  // Start by setting the first pose at the global origin
  steam::se3::TransformStateVar::Ptr temp(new steam::se3::TransformStateVar(dataset.frames_ic[0].T_k0));
  relposes_ic_k_kp.push_back(temp);
  relposes_ic_k_kp[0]->setLock(true);

  // Add all the relative poses
  for (unsigned int i = 1; i < poses_ic_k_0.size(); i++) {

    // get references to the current and previous poses
    steam::se3::TransformStateVar::Ptr& c_poseVar = poses_ic_k_0[i];
    steam::se3::TransformStateVar::Ptr& cp_poseVar = poses_ic_k_0[i-1];

    // get the relative transform between the current and previous pose
    steam::se3::TransformStateVar::Ptr transform_vk_vkp(new steam::se3::TransformStateVar(c_poseVar->getValue()/cp_poseVar->getValue()));
    relposes_ic_k_kp.push_back(transform_vk_vkp);
  }

  // make a set of transform evaluators that correspond to each relative pose
  std::vector<steam::se3::TransformEvaluator::Ptr> transform_evals_ic_k_kp;

  // start by inserting the origin pose
  steam::se3::TransformEvaluator::Ptr pose_vk_vk0 = steam::se3::TransformStateEvaluator::MakeShared(relposes_ic_k_kp[0]);
  transform_evals_ic_k_kp.push_back(pose_vk_vk0);

  // Add all the relative transforms
  for (unsigned int i = 1; i < relposes_ic_k_kp.size(); i++) {

  // get the relative pose transform
    steam::se3::TransformEvaluator::Ptr transform_eval_vk_vkp = steam::se3::TransformStateEvaluator::MakeShared(relposes_ic_k_kp[i]);

    // update the transform and save it
    steam::se3::TransformEvaluator::Ptr pose_vk_vkp = steam::se3::compose(transform_eval_vk_vkp, transform_evals_ic_k_kp[i-1]);
    transform_evals_ic_k_kp.push_back(pose_vk_vkp);
  }

  ///
  /// Setup Cost Terms
  ///

  // steam cost terms
  std::vector<steam::CostTerm::Ptr> costTerms;

  // Setup shared noise and loss function
  steam::NoiseModel::Ptr sharedCameraNoiseModel(new steam::NoiseModel(dataset.noise));
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Setup camera intrinsics
  steam::StereoCameraErrorEval::CameraIntrinsics::Ptr sharedIntrinsics(
        new steam::StereoCameraErrorEval::CameraIntrinsics());
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
    //steam::se3::TransformStateVar::Ptr& poseVar = relposes_ic_k_kp[frameIdx];


    // Setup landmark if first time
    unsigned int landmarkIdx = dataset.meas[i].landID;
    if (!landmarks_ic[landmarkIdx]) {
      Eigen::Vector4d p_v0; p_v0.head<3>() = dataset.land_ic[landmarkIdx].point; p_v0[3] = 1.0;
      Eigen::Vector4d p_vl = (poses_ic_k_0[frameIdx]->getValue()/poses_ic_k_0[0]->getValue()) * p_v0;
      landmarks_ic[landmarkIdx] = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p_vl.head<3>(), transform_evals_ic_k_kp[frameIdx]));
    }

    // Get landmark reference
    steam::se3::LandmarkStateVar::Ptr& landVar = landmarks_ic[landmarkIdx];

    // Construct transform evaluator between landmark frame (inertial) and camera frame
    steam::se3::TransformEvaluator::Ptr pose_c_cp = steam::se3::compose(pose_c_v, transform_evals_ic_k_kp[frameIdx]);

    // Construct error function
    steam::StereoCameraErrorEval::Ptr errorfunc(new steam::StereoCameraErrorEval(
            dataset.meas[i].data, sharedIntrinsics, pose_c_cp, landVar));

    // Construct cost term
    steam::CostTerm::Ptr cost(new steam::CostTerm(errorfunc, sharedCameraNoiseModel, sharedLossFunc));
    costTerms.push_back(cost);
  }

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add pose variables
  for (unsigned int i = 1; i < relposes_ic_k_kp.size(); i++) {
    problem.addStateVariable(relposes_ic_k_kp[i]);
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

  // Initialize parameters (enable verbose mode)
  steam::DoglegGaussNewtonSolver::Params params;
  params.verbose = true;

  // Make solver
  steam::DoglegGaussNewtonSolver solver(&problem, params);

  // Optimize
  solver.optimize();

  return 0;
}
