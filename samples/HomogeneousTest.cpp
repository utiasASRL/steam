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
#include <steam/evaluator/samples/RangeConditioningEval.hpp>

Eigen::Vector4d stereoModel(const Eigen::Vector4d& point, const steam::StereoCameraErrorEval::CameraIntrinsics::Ptr& intrinsics_) {

  // Precompute values
  const double x = point[0];
  const double y = point[1];
  const double z = point[2];
  const double w = point[3];
  const double xr = x - w * intrinsics_->b;
  const double one_over_z = 1.0/z;

  // Project point into camera coordinates
  Eigen::Vector4d projectedMeas;
  projectedMeas << intrinsics_->fu *  x  * one_over_z + intrinsics_->cu,
                   intrinsics_->fv *  y  * one_over_z + intrinsics_->cv,
                   intrinsics_->fu *  xr * one_over_z + intrinsics_->cu,
                   intrinsics_->fv *  y  * one_over_z + intrinsics_->cv;
  return projectedMeas;
}

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
    std::cout << "Parsing default file: " << filename << std::endl << std::endl;
  } else {
    filename = argv[1];
    std::cout << "Parsing file: " << filename << std::endl << std::endl;
  }

  // Load dataset
  steam::data::SimpleBaDataset dataset = steam::data::parseSimpleBaDataset(filename);

  ///
  /// Setup and Initialize States
  ///

  // State variable containers (and related data)
  steam::se3::TransformStateVar::Ptr pose1(new steam::se3::TransformStateVar(lgmath::se3::Transformation()));
  pose1->setLock(true);
  Eigen::Matrix<double,6,1> xi = 0.1*Eigen::Matrix<double,6,1>::Random();
  steam::se3::TransformStateVar::Ptr pose2(new steam::se3::TransformStateVar(lgmath::se3::Transformation(xi)));

  std::vector<steam::se3::LandmarkStateVar::Ptr> landmarks_gt;
  std::vector<steam::se3::LandmarkStateVar::Ptr> landmarks_ic;

  // Setup landmarks

  Eigen::Vector3d p;
  steam::se3::LandmarkStateVar::Ptr newLandGt;
  steam::se3::LandmarkStateVar::Ptr newLandIc;

  double c = 0.01;
  p << c, 0.0, 0.0;
  newLandGt = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p));
  newLandIc = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p+0.2*c*Eigen::Vector3d::Random()));
  landmarks_gt.push_back(newLandGt);
  landmarks_ic.push_back(newLandIc);

  c = 1.0;
  p << c, 0.0, 0.0;
  newLandGt = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p));
  newLandIc = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p+0.2*c*Eigen::Vector3d::Random()));
  landmarks_gt.push_back(newLandGt);
  landmarks_ic.push_back(newLandIc);

  c = 100000.0;
  p << c, 0.0, 0.0;
  newLandGt = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p));
  newLandIc = steam::se3::LandmarkStateVar::Ptr(new steam::se3::LandmarkStateVar(p+0.2*c*Eigen::Vector3d::Random()));
  landmarks_gt.push_back(newLandGt);
  landmarks_ic.push_back(newLandIc);


  ///
  /// Setup Cost Terms
  ///

  // steam cost terms
  steam::CostTermCollection<4,6>::Ptr stereoCostTerms(new steam::CostTermCollection<4,6>());
  steam::CostTermCollection<1,3>::Ptr rangeCostTerms(new steam::CostTermCollection<1,3>());

  // Setup shared noise and loss function
  steam::NoiseModel<4>::Ptr sharedCameraNoiseModel(new steam::NoiseModel<4>(dataset.noise));
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Setup camera intrinsics
  steam::StereoCameraErrorEval::CameraIntrinsics::Ptr sharedIntrinsics(
        new steam::StereoCameraErrorEval::CameraIntrinsics());
  sharedIntrinsics->b  = dataset.camParams.b;
  sharedIntrinsics->fu = dataset.camParams.fu;
  sharedIntrinsics->fv = dataset.camParams.fv;
  sharedIntrinsics->cu = dataset.camParams.cu;
  sharedIntrinsics->cv = dataset.camParams.cv;

  // Generate cost terms for camera measurements
  for (unsigned int i = 0; i < landmarks_ic.size(); i++) {

    // Get landmark reference
    steam::se3::LandmarkStateVar::Ptr& landVar = landmarks_ic[i];


    {
      /*if (i == 2) {

        steam::NoiseModel<1>::Ptr sharedRangeModel(new steam::NoiseModel<1>(1000*Eigen::Matrix<double,1,1>::Identity()));

        // Construct error function
        steam::RangeConditioningEval::Ptr errorfunc(new steam::RangeConditioningEval(landVar));

        // Construct cost term
        steam::CostTerm<1,3>::Ptr cost(new steam::CostTerm<1,3>(errorfunc, sharedRangeModel, sharedLossFunc));
        rangeCostTerms->add(cost);
      }*/
    }

    {
      // Construct transform evaluator between landmark frame (inertial) and camera frame
      steam::se3::TransformEvaluator::Ptr pose_c_v = steam::se3::FixedTransformEvaluator::MakeShared(dataset.T_cv);
      steam::se3::TransformEvaluator::Ptr pose_vk_0 = steam::se3::TransformStateEvaluator::MakeShared(pose1);
      steam::se3::TransformEvaluator::Ptr pose_c_0 = steam::se3::compose(pose_c_v, pose_vk_0);

      Eigen::Vector4d meas = stereoModel(pose_c_0->evaluate()*landmarks_gt[i]->getValue(), sharedIntrinsics) + 0.1*Eigen::Vector4d::Random();
      std::cout << meas[0] - meas[2] << std::endl;

      // Construct error function
      steam::StereoCameraErrorEval::Ptr errorfunc(new steam::StereoCameraErrorEval(
              meas, sharedIntrinsics, pose_c_0, landVar));

      // Construct cost term
      steam::CostTerm<4,6>::Ptr cost(new steam::CostTerm<4,6>(errorfunc, sharedCameraNoiseModel, sharedLossFunc));
      stereoCostTerms->add(cost);
    }

    {
      // Construct transform evaluator between landmark frame (inertial) and camera frame
      steam::se3::TransformEvaluator::Ptr pose_c_v = steam::se3::FixedTransformEvaluator::MakeShared(dataset.T_cv);
      steam::se3::TransformEvaluator::Ptr pose_vk_0 = steam::se3::TransformStateEvaluator::MakeShared(pose2);
      steam::se3::TransformEvaluator::Ptr pose_c_0 = steam::se3::compose(pose_c_v, pose_vk_0);

      Eigen::Vector4d meas = stereoModel(pose_c_0->evaluate()*landmarks_gt[i]->getValue(), sharedIntrinsics) + 0.1*Eigen::Vector4d::Random();
      std::cout << meas[0] - meas[2] << std::endl;

      // Construct error function
      steam::StereoCameraErrorEval::Ptr errorfunc(new steam::StereoCameraErrorEval(
              meas, sharedIntrinsics, pose_c_0, landVar));

      // Construct cost term
      steam::CostTerm<4,6>::Ptr cost(new steam::CostTerm<4,6>(errorfunc, sharedCameraNoiseModel, sharedLossFunc));
      stereoCostTerms->add(cost);
    }
  }

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  problem.addStateVariable(pose2);

  // Add landmark variables
  for (unsigned int i = 0; i < landmarks_ic.size(); i++) {
    problem.addStateVariable(landmarks_ic[i]);
  }

  // Add cost terms
  problem.addCostTermCollection(stereoCostTerms);
  problem.addCostTermCollection(rangeCostTerms);

  ///
  /// Setup Solver and Optimize
  ///
  typedef steam::DoglegGaussNewtonSolver SolverType;
  //typedef steam::LineSearchGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  solver.optimize();

  return 0;
}
