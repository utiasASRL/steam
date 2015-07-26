//////////////////////////////////////////////////////////////////////////////////////////////
/// \file NaiveGPandBA.cpp
/// \brief A sample usage of the STEAM Engine library for a bundle adjustment problem with a
///        Gaussian process prior.
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <steam/data/ParseBA.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief A naive implementation of the GP prior
//////////////////////////////////////////////////////////////////////////////////////////////
class NaiveGPFactorEval : public ErrorEvaluator
{
public:

  /// Shared pointer typedefs for readability
  typedef boost::shared_ptr<NaiveGPFactorEval> Ptr;
  typedef boost::shared_ptr<const NaiveGPFactorEval> ConstPtr;

  /// Constructor
  NaiveGPFactorEval(double time1, double time2,
                    const se3::TransformStateVar::ConstPtr& pose_T_10,
                    const se3::TransformStateVar::ConstPtr& pose_T_20,
                    const VectorSpaceStateVar::ConstPtr& varpi1,
                    const VectorSpaceStateVar::ConstPtr& varpi2)
    : time1_(time1), time2_(time2), pose_T_10_(pose_T_10), pose_T_20_(pose_T_20),
      varpi1_(varpi1), varpi2_(varpi2) {
  }

  /// \brief Returns whether or not an evaluator contains unlocked state variables
  virtual bool isActive() const {
    return !pose_T_10_->isLocked() ||
           !pose_T_20_->isLocked() ||
           !varpi1_->isLocked() ||
           !varpi2_->isLocked();
  }

  /// Evaluate the measurement error
  virtual Eigen::VectorXd evaluate() const {
    lgmath::se3::Transformation T_21 = pose_T_20_->getValue()/pose_T_10_->getValue();
    Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
    Eigen::Matrix<double,6,6> Jinv_21 = lgmath::se3::vec2jacinv(xi_21);

    Eigen::Matrix<double,12,1> error;
    error.head<6>() = xi_21 - (time2_ - time1_)*varpi1_->getValue();
    error.tail<6>() = Jinv_21 * varpi2_->getValue() - varpi1_->getValue();
    return error;
  }

  virtual Eigen::VectorXd evaluate(std::vector<Jacobian>* jacs) const {

    lgmath::se3::Transformation T_21 = pose_T_20_->getValue()/pose_T_10_->getValue();
    Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
    Eigen::Matrix<double,6,6> Jinv_21 = lgmath::se3::vec2jacinv(xi_21);
    Eigen::Matrix<double,6,6> Jinv_12 = Jinv_21*T_21.adjoint();
    double deltaTime = (time2_ - time1_);

    // Check and initialize jacobian array
    if (jacs == NULL) {
      throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
    }
    jacs->clear();
    jacs->reserve(4);

    if(!pose_T_10_->isLocked()) {
      jacs->push_back(Jacobian());
      size_t i = jacs->size() - 1;
      (*jacs)[i].key = pose_T_10_->getKey();
      (*jacs)[i].jac = Eigen::Matrix<double,12,6>();
      (*jacs)[i].jac.block<6,6>(0,0) = -Jinv_12;
      (*jacs)[i].jac.block<6,6>(6,0) = -0.5*lgmath::se3::curlyhat(varpi2_->getValue())*Jinv_12;
    }

    if(!varpi1_->isLocked()) {
      jacs->push_back(Jacobian());
      size_t i = jacs->size() - 1;
      (*jacs)[i].key = varpi1_->getKey();
      (*jacs)[i].jac = Eigen::Matrix<double,12,6>();
      (*jacs)[i].jac.block<6,6>(0,0) = -deltaTime*Eigen::Matrix<double,6,6>::Identity();
      (*jacs)[i].jac.block<6,6>(6,0) = -Eigen::Matrix<double,6,6>::Identity();
    }

    if(!pose_T_20_->isLocked()) {
      jacs->push_back(Jacobian());
      size_t i = jacs->size() - 1;
      (*jacs)[i].key = pose_T_20_->getKey();
      (*jacs)[i].jac = Eigen::Matrix<double,12,6>();
      (*jacs)[i].jac.block<6,6>(0,0) = Jinv_21;
      (*jacs)[i].jac.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(varpi2_->getValue())*Jinv_21;
    }

    if(!varpi2_->isLocked()) {
      jacs->push_back(Jacobian());
      size_t i = jacs->size() - 1;
      (*jacs)[i].key = varpi2_->getKey();
      (*jacs)[i].jac = Eigen::Matrix<double,12,6>();
      (*jacs)[i].jac.block<6,6>(0,0) = Eigen::Matrix<double,6,6>::Zero();
      (*jacs)[i].jac.block<6,6>(6,0) = Jinv_21;
    }

    // Return error
    Eigen::Matrix<double,12,1> error;
    error.head<6>() = xi_21 - deltaTime*varpi1_->getValue();
    error.tail<6>() = Jinv_21 * varpi2_->getValue() - varpi1_->getValue();
    return error;
  }

private:

  double time1_; // TODO time should not be double...
  double time2_;
  se3::TransformStateVar::ConstPtr pose_T_10_;
  se3::TransformStateVar::ConstPtr pose_T_20_;
  VectorSpaceStateVar::ConstPtr varpi1_;
  VectorSpaceStateVar::ConstPtr varpi2_;

};

} // steam

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves simple bundle adjustment problems, with a GP prior
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  ///
  /// Parse Dataset - sphere of relative pose measurements (fairly dense loop closures)
  ///

  // Get filename
  std::string filename;
  if (argc < 2) {
    filename = "../../include/steam/data/stereo_dataset3_1000.txt";
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
  std::vector<double> times_ic;
  std::vector<steam::se3::TransformStateVar::Ptr> poses_ic_k_0;
  std::vector<steam::VectorSpaceStateVar::Ptr> varpi_ic;
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
    times_ic.push_back(dataset.frames_ic[i].time);
  }

  // Lock first pose (otherwise entire solution is 'floating')
  //  **Note: alternatively we could add a prior (UnaryTransformError) to the first pose.
  poses_ic_k_0[0]->setLock(true);

  // Setup velocities based on initial pose conditions
  for (unsigned int i = 1; i < poses_ic_k_0.size(); i++) {
    double deltaTime = times_ic[i]-times_ic[i-1];
    Eigen::Matrix<double,6,1> varpi = (1.0/deltaTime) * (poses_ic_k_0[i]->getValue()/poses_ic_k_0[i-1]->getValue()).vec();
    steam::VectorSpaceStateVar::Ptr temp(new steam::VectorSpaceStateVar(varpi));
    varpi_ic.push_back(temp);
    if (i+1 == poses_ic_k_0.size()) {
      steam::VectorSpaceStateVar::Ptr temp2(new steam::VectorSpaceStateVar(varpi));
      varpi_ic.push_back(temp2);
    }
  }

  // Setup landmarks with initial condition
  for (unsigned int i = 0; i < dataset.land_ic.size(); i++) {
    steam::se3::LandmarkStateVar::Ptr temp(new steam::se3::LandmarkStateVar(dataset.land_ic[i].point));
    landmarks_ic.push_back(temp);
  }

  ///
  /// Setup Cost Terms
  ///

  // steam cost terms
  std::vector<steam::CostTerm::Ptr> costTerms;

  // Setup shared noise and loss function
  steam::NoiseModel::Ptr sharedCameraNoiseModel(new steam::NoiseModel(dataset.noise));
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Unary velocity prior
  //   **Note: may not be required if there is adequate measurements, but is
  //           needed to make a standalone prior that is well conditioned
  {
    // Setup noise for initial velocity (very uncertain)
    steam::NoiseModel::Ptr sharedInitialVelocityNoiseModel(new steam::NoiseModel(1000.0*Eigen::MatrixXd::Identity(6,6)));

    // Setup zero measurement
    steam::VectorSpaceStateVar::Ptr& varpiVar = varpi_ic[0];
    Eigen::VectorXd meas = Eigen::Matrix<double,6,1>::Zero();

    // Setup unary error and cost term
    steam::VectorSpaceErrorEval::Ptr errorfunc(new steam::VectorSpaceErrorEval(meas, varpiVar));
    steam::CostTerm::Ptr cost(new steam::CostTerm(errorfunc, sharedInitialVelocityNoiseModel, sharedLossFunc));
    costTerms.push_back(cost);
  }

  // Generate cost terms related to GP prior
  for (unsigned int i = 1; i < poses_ic_k_0.size(); i++)
  {
    // Get references to times, poses and velocities
    double timeA = times_ic[i-1];
    double timeB = times_ic[i];
    steam::se3::TransformStateVar::Ptr& poseVarA = poses_ic_k_0[i-1];
    steam::se3::TransformStateVar::Ptr& poseVarB = poses_ic_k_0[i];
    steam::VectorSpaceStateVar::Ptr& varpiVarA = varpi_ic[i-1];
    steam::VectorSpaceStateVar::Ptr& varpiVarB = varpi_ic[i];

    // Setup constant portion of GP prior factor
    Eigen::Array<double,1,6> Qc_diag; Qc_diag << 0.1012,    0.0935,    0.0258,    0.2392,    0.6359,    0.7239;
    Eigen::Matrix<double,6,6> Qc_inv = Eigen::Matrix<double,6,6>::Zero();
    Qc_inv.diagonal() = 1.0/Qc_diag;

    // Generate 12 x 12 covariance/information matrix for GP prior factor
    Eigen::Matrix<double,12,12> Qi_inv;
    double one_over_dt = 1.0/(timeB - timeA);
    double one_over_dt2 = one_over_dt*one_over_dt;
    double one_over_dt3 = one_over_dt2*one_over_dt;
    Qi_inv.block<6,6>(0,0) = 12.0 * one_over_dt3 * Qc_inv;
    Qi_inv.block<6,6>(6,0) = Qi_inv.block<6,6>(0,6) = -6.0 * one_over_dt2 * Qc_inv;
    Qi_inv.block<6,6>(6,6) =  4.0 * one_over_dt  * Qc_inv;
    steam::NoiseModel::Ptr sharedGPNoiseModel(new steam::NoiseModel(Qi_inv, steam::NoiseModel::INFORMATION));

    // Create cost term
    steam::NaiveGPFactorEval::Ptr errorfunc(new steam::NaiveGPFactorEval(timeA, timeB, poseVarA, poseVarB, varpiVarA, varpiVarB));
    steam::CostTerm::Ptr cost(new steam::CostTerm(errorfunc, sharedGPNoiseModel, sharedLossFunc));
    costTerms.push_back(cost);
  }

  // Setup camera intrinsics
  steam::StereoCameraErrorEval::CameraIntrinsics::Ptr sharedIntrinsics(
        new steam::StereoCameraErrorEval::CameraIntrinsics());
  sharedIntrinsics->b  = dataset.camParams.b;
  sharedIntrinsics->fu = dataset.camParams.fu;
  sharedIntrinsics->fv = dataset.camParams.fv;
  sharedIntrinsics->cu = dataset.camParams.cu;
  sharedIntrinsics->cv = dataset.camParams.cv;

  // Generate cost terms for camera measurements
  for (unsigned int i = 0; i < dataset.meas.size(); i++) {

    // Get pose reference
    steam::se3::TransformStateVar::Ptr& poseVar = poses_ic_k_0[dataset.meas[i].frameID];

    // Get landmark reference
    steam::se3::LandmarkStateVar::Ptr& landVar = landmarks_ic[dataset.meas[i].landID];

    // Construct transform evaluator between landmark frame (inertial) and camera frame
    steam::se3::TransformEvaluator::Ptr pose_c_v = steam::se3::FixedTransformEvaluator::MakeShared(dataset.T_cv);
    steam::se3::TransformEvaluator::Ptr pose_v_0 = steam::se3::TransformStateEvaluator::MakeShared(poseVar);
    steam::se3::TransformEvaluator::Ptr pose_c_0 = steam::se3::compose(pose_c_v, pose_v_0);

    // Construct error function
    steam::StereoCameraErrorEval::Ptr errorfunc(new steam::StereoCameraErrorEval(
            dataset.meas[i].data, sharedIntrinsics, pose_c_0, landVar));

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
  for (unsigned int i = 1; i < poses_ic_k_0.size(); i++) {
    problem.addStateVariable(poses_ic_k_0[i]);
  }

  // Add velocity variables
  for (unsigned int i = 0; i < varpi_ic.size(); i++) {
    problem.addStateVariable(varpi_ic[i]);
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
