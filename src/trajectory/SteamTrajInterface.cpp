//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajInterface.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/SteamTrajInterface.hpp>

#include <lgmath.hpp>

#include <steam/trajectory/SteamTrajPoseInterpEval.hpp>
#include <steam/trajectory/SteamTrajPriorFactor.hpp>
#include <steam/evaluator/samples/VectorSpaceErrorEval.hpp>

#include <steam/evaluator/blockauto/transform/TransformEvalOperations.hpp>
#include <steam/evaluator/blockauto/transform/ConstVelTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
///        Note, without providing Qc, the trajectory can be used safely for interpolation,
///        but should not be used for estimation.
//////////////////////////////////////////////////////////////////////////////////////////////
SteamTrajInterface::SteamTrajInterface(bool allowExtrapolation) :
  Qc_inv_(Eigen::Matrix<double,6,6>::Identity()), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamTrajInterface::SteamTrajInterface(const Eigen::Matrix<double,6,6>& Qc_inv,
                                       bool allowExtrapolation) :
  Qc_inv_(Qc_inv), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::add(const SteamTrajVar::Ptr& knot) {

  // Todo, check that time does not already exist in map?

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamTrajVar::Ptr>(knot->getTime().nanosecs(), knot));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::add(const steam::Time& time,
                             const se3::TransformEvaluator::Ptr& T_k0,
                             const VectorSpaceStateVar::Ptr& velocity) {

  // Check velocity input
  if (velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }

  // Todo, check that time does not already exist in map?

  // Make knot
  SteamTrajVar::Ptr newEntry(new SteamTrajVar(time, T_k0, velocity));

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamTrajVar::Ptr>(time.nanosecs(), newEntry));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
TransformEvaluator::ConstPtr SteamTrajInterface::getInterpPoseEval(const steam::Time& time) const {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      --it1; // should be safe, as we checked that the map was not empty..
      const SteamTrajVar::Ptr& endKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(endKnot->getVelocity(), time - endKnot->getTime());
      return compose(T_t_k, endKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {

    // return state variable exactly (no interp)
    return it1->second->getPose();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      const SteamTrajVar::Ptr& startKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(startKnot->getVelocity(),
                                                 time - startKnot->getTime());
      return compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // Create interpolated evaluator
  return SteamTrajPoseInterpEval::MakeShared(time, it1->second, it2->second);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
///        should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::addPosePrior(const steam::Time& time,
                                      const lgmath::se3::Transformation& pose,
                                      const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] no knot at provided time.");
  }

  // Get reference
  const SteamTrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(!knotRef->getPose()->isActive()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] tried to add prior to locked pose.");
  }

  // Set up loss function, noise model, and error function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(cov));
  steam::TransformErrorEval::Ptr errorfunc(new steam::TransformErrorEval(pose, knotRef->getPose()));

  // Create cost term
  posePriorFactor_ = steam::WeightedLeastSqCostTerm<6,6>::Ptr(
        new steam::WeightedLeastSqCostTerm<6,6>(errorfunc, sharedNoiseModel, sharedLossFunc));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary velocity prior factor at a knot time. Note that only a single velocity
///        prior should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::addVelocityPrior(const steam::Time& time,
                                          const Eigen::Matrix<double,6,1>& velocity,
                                          const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const SteamTrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(knotRef->getVelocity()->isLocked()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] tried to add prior to locked pose.");
  }

  // Set up loss function, noise model, and error function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(cov));
  steam::VectorSpaceErrorEval<6,6>::Ptr errorfunc(new steam::VectorSpaceErrorEval<6,6>(velocity, knotRef->getVelocity()));

  // Create cost term
  velocityPriorFactor_ = steam::WeightedLeastSqCostTerm<6,6>::Ptr(
        new steam::WeightedLeastSqCostTerm<6,6>(errorfunc, sharedNoiseModel, sharedLossFunc));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get cost terms associated with the prior for unlocked parts of the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::appendPriorCostTerms(
    const ParallelizedCostTermCollection::Ptr& costTerms) const {

  // If empty, return none
  if (knotMap_.empty()) {
    return;
  }

  // Check for pose or velocity priors
  if (posePriorFactor_) {
    costTerms->add(posePriorFactor_);
  }
  if (velocityPriorFactor_) {
    costTerms->add(velocityPriorFactor_);
  }

  // All prior factors will use an L2 loss function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Initialize first iterator
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1 = knotMap_.begin();
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("No knots...");
  }

  // Iterate through all states.. if any are unlocked, supply a prior term
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it2 = it1; ++it2;
  for (; it2 != knotMap_.end(); ++it1, ++it2) {

    // Get knots
    const SteamTrajVar::ConstPtr& knot1 = it1->second;
    const SteamTrajVar::ConstPtr& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if(knot1->getPose()->isActive()  || !knot1->getVelocity()->isLocked() ||
       knot2->getPose()->isActive()  || !knot2->getVelocity()->isLocked() ) {

      // Generate 12 x 12 information matrix for GP prior factor
      Eigen::Matrix<double,12,12> Qi_inv;
      double one_over_dt = 1.0/(knot2->getTime() - knot1->getTime()).seconds();
      double one_over_dt2 = one_over_dt*one_over_dt;
      double one_over_dt3 = one_over_dt2*one_over_dt;
      Qi_inv.block<6,6>(0,0) = 12.0 * one_over_dt3 * Qc_inv_;
      Qi_inv.block<6,6>(6,0) = Qi_inv.block<6,6>(0,6) = -6.0 * one_over_dt2 * Qc_inv_;
      Qi_inv.block<6,6>(6,6) = 4.0 * one_over_dt  * Qc_inv_;
      steam::BaseNoiseModelX::Ptr sharedGPNoiseModel(
            new steam::StaticNoiseModelX(Qi_inv, steam::INFORMATION));

      // Create cost term
      steam::se3::SteamTrajPriorFactor::Ptr errorfunc(
            new steam::se3::SteamTrajPriorFactor(knot1, knot2));
      steam::WeightedLeastSqCostTermX::Ptr cost(
            new steam::WeightedLeastSqCostTermX(errorfunc, sharedGPNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  // Iterate over trajectory
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {

    // Append active states in transform evaluator
    it->second->getPose()->getActiveStateVariables(outStates);

    // Check if velocity is locked
    if (!it->second->getVelocity()->isLocked()) {
      (*outStates)[it->second->getVelocity()->getKey().getID()] = it->second->getVelocity();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get interpolated/extrapolated covariance at given time
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::getInterpCov(GaussNewtonSolverBase& solver,
    const steam::Time& time) const {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // TODO(DAVID): Extrapolation past last entry
  // Check if time is passed the last entry
      /*
  if (it1 == knotMap_.end()) {

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      --it1; // should be safe, as we checked that the map was not empty..
      const SteamTrajVar::Ptr& endKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(endKnot->getVelocity(), time - endKnot->getTime());
      return compose(T_t_k, endKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }
  */

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
    // return covariance exactly (no interp)
    std::map<unsigned int, steam::StateVariableBase::Ptr> outState;
    it1->second->getPose()->getActiveStateVariables(&outState);
    
    std::vector<steam::StateKey> keys;
    keys.push_back(outState.begin()->second->getKey());
    keys.push_back(it1->second->getVelocity()->getKey());

    steam::BlockMatrix covariance = solver.queryCovarianceBlock(keys);

    Eigen::Matrix<double,12,12> output;
    output.block<6,6>(0,0) = covariance.copyAt(0,0);
    output.block<6,6>(0,6) = covariance.copyAt(0,1);
    output.block<6,6>(6,0) = covariance.copyAt(1,0);
    output.block<6,6>(6,6) = covariance.copyAt(1,1);
    return output;
  }

  // TODO(DAVID): Extrapolation behind first entry
  /*
  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      const SteamTrajVar::Ptr& startKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(startKnot->getVelocity(),
                                                 time - startKnot->getTime());
      return compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }
  */

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // Need interpolated pose
  TransformEvaluator::ConstPtr interp_state = SteamTrajPoseInterpEval::MakeShared(
      time, it1->second, it2->second);

  // Get required pose/velocity keys
  // TODO(DAVID): Find a better way to get pose keys...
  std::vector<steam::StateKey> keys;
  {
    std::map<unsigned int, steam::StateVariableBase::Ptr> outState1;
    it1->second->getPose()->getActiveStateVariables(&outState1);

    std::map<unsigned int, steam::StateVariableBase::Ptr> outState2;
    it2->second->getPose()->getActiveStateVariables(&outState2);

    keys.push_back(outState1.begin()->second->getKey());
    keys.push_back(it1->second->getVelocity()->getKey());
    keys.push_back(outState2.begin()->second->getKey());
    keys.push_back(it2->second->getVelocity()->getKey());
  }

  // Get required covariances using the keys
  steam::BlockMatrix global_cov = solver.queryCovarianceBlock(keys);

  // Approximately translate global covariances (P_11 ... P_22) to local frame 1
  Eigen::Matrix<double,24,24> local_cov = translateCovToLocal(global_cov,
      it1->second, it2->second);

  // Interpolate using local covariances
  double psi11, psi12, psi21, psi22, lambda11, lambda12, lambda21, lambda22;
  { // TODO(DAVID): Copied from SteamTrajPoseInterpEval. Look into a better way.
    double tau = (time - it1->second->getTime()).seconds();
    double T = (it2->second->getTime() - it1->second->getTime()).seconds();
    double ratio = tau/T;
    double ratio2 = ratio*ratio;
    double ratio3 = ratio2*ratio;

    // Calculate 'psi' interpolation values
    psi11 = 3.0*ratio2 - 2.0*ratio3;
    psi12 = tau*(ratio2 - ratio);
    psi21 = 6.0*(ratio - ratio2)/T;
    psi22 = 3.0*ratio2 - 2.0*ratio;

    // Calculate 'lambda' interpolation values
    lambda11 = 1.0 - psi11;
    lambda12 = tau - T*psi11 - psi12;
    lambda21 = -psi21;
    lambda22 = 1.0 - psi21 - psi22;
  }

  // Approximately translate result to global frame

  // Dummy return
  return local_cov;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Covariance translation
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::translateCovToLocal(const steam::BlockMatrix& global_cov, 
    const SteamTrajVar::ConstPtr& knot1, const SteamTrajVar::ConstPtr& knot2) const {

  Eigen::Matrix<double,12,12> P_11;
  Eigen::Matrix<double,12,12> P_12;
  Eigen::Matrix<double,12,12> P_22;

  P_11.block<6,6>(0,0) = global_cov.copyAt(0,0);
  P_11.block<6,6>(0,6) = global_cov.copyAt(0,1);
  P_11.block<6,6>(6,0) = global_cov.copyAt(1,0);
  P_11.block<6,6>(6,6) = global_cov.copyAt(1,1);

  P_12.block<6,6>(0,0) = global_cov.copyAt(0,2);
  P_12.block<6,6>(0,6) = global_cov.copyAt(0,3);
  P_12.block<6,6>(6,0) = global_cov.copyAt(1,2);
  P_12.block<6,6>(6,6) = global_cov.copyAt(1,3);

  P_22.block<6,6>(0,0) = global_cov.copyAt(2,2);
  P_22.block<6,6>(0,6) = global_cov.copyAt(2,3);
  P_22.block<6,6>(6,0) = global_cov.copyAt(3,2);
  P_22.block<6,6>(6,6) = global_cov.copyAt(3,3);

  // Create constants
  Eigen::Matrix<double,12,12> Gam1 = Eigen::MatrixXd::Zero(12,12);
  Eigen::Matrix<double,12,12> Gam2 = Eigen::MatrixXd::Zero(12,12);
  Eigen::Matrix<double,12,12> Xi1 = Eigen::MatrixXd::Zero(12,12);
  Eigen::Matrix<double,12,12> Xi2 = Eigen::MatrixXd::Zero(12,12);

  // Eigen::Matrix4d identity_transform = Eigen::MatrixXd::Identity(4,4);
  // lgmath::se3::Transformation T_11(identity_transform);
  // Eigen::Matrix<double,6,6> J_11_inv = lgmath::se3::vec2jacinv(T_11.vec());
  Eigen::Matrix<double,6,6> J_11_inv = Eigen::MatrixXd::Identity(6,6);
  Gam1.block<6,6>(0,0) = J_11_inv;
  Gam1.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(knot1->getVelocity()->getValue());
  Gam1.block<6,6>(6,6) = J_11_inv;

  lgmath::se3::Transformation T_21 = knot2->getPose()->evaluate()/knot1->getPose()->evaluate();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(T_21.vec());
  Gam2.block<6,6>(0,0) = J_21_inv;
  Gam2.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(knot2->getVelocity()->getValue())*J_21_inv;
  Gam2.block<6,6>(6,6) = J_21_inv;

  Xi1.block<6,6>(0,0) = Eigen::MatrixXd::Identity(6,6);
  Xi2.block<6,6>(0,0) = lgmath::se3::tranAd(T_21.matrix());

  // Translate
  Eigen::Matrix<double,12,12> P_11_local = Gam1*(P_11 - Xi1*P_11*Xi1.transpose())*Gam1.transpose();
  Eigen::Matrix<double,12,12> P_12_local = Gam1*(P_12 - Xi1*P_11*Xi2.transpose())*Gam2.transpose();
  Eigen::Matrix<double,12,12> P_22_local = Gam2*(P_22 - Xi2*P_11*Xi2.transpose())*Gam2.transpose();

  // Return
  Eigen::Matrix<double,24,24> output;
  output.block<12,12>(0,0)   = P_11_local;
  output.block<12,12>(0,12)  = P_12_local;
  output.block<12,12>(12,0)  = P_12_local.transpose();
  output.block<12,12>(12,12) = P_22_local;
  return output;
}

} // se3
} // steam
