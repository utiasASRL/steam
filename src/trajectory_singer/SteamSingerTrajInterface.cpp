//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamSingerTrajInterface.cpp
///
/// \author Jeremy Wong, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory_singer/SteamSingerTrajInterface.hpp>

#include <lgmath.hpp>
// #include <unsupported/Eigen/MatrixFunctions>

#include <steam/trajectory_singer/SteamSingerTrajPoseInterpEval.hpp>
#include <steam/trajectory_singer/SteamSingerTrajPriorFactor.hpp>
#include <steam/evaluator/samples/VectorSpaceErrorEval.hpp>

#include <steam/evaluator/blockauto/transform/TransformEvalOperations.hpp>
#include <steam/evaluator/blockauto/transform/ConstVelTransformEvaluator.hpp>
#include <steam/evaluator/blockauto/transform/SingerTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
///        Note, without providing Qc, the trajectory can be used safely for interpolation,
///        but should not be used for estimation.
//////////////////////////////////////////////////////////////////////////////////////////////
SteamSingerTrajInterface::SteamSingerTrajInterface(bool allowExtrapolation) :
  Qc_(Eigen::Matrix<double,6,6>::Identity()), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamSingerTrajInterface::SteamSingerTrajInterface(const Eigen::Matrix<double,6,6>& Qc, const Eigen::Matrix<double,6,6>& alpha, const Eigen::Matrix<double,6,6>& alpha_inv,
                                       bool allowExtrapolation) :
  Qc_(Qc), alpha_(alpha), alpha_inv_(alpha_inv), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajInterface::add(const SteamSingerTrajVar::Ptr& knot) {

  // Todo, check that time does not already exist in map?

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamSingerTrajVar::Ptr>(knot->getTime().nanosecs(), knot));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajInterface::add(const steam::Time& time,
                             const se3::TransformEvaluator::Ptr& T_k0,
                             const VectorSpaceStateVar::Ptr& velocity,
                             const VectorSpaceStateVar::Ptr& acceleration) {

  // Check velocity input
  if (velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }

  // Check acceleration input
  if (acceleration->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid acceleration size");
  }

  // Todo, check that time does not already exist in map?

  // Make knot
  SteamSingerTrajVar::Ptr newEntry(new SteamSingerTrajVar(time, T_k0, velocity, acceleration));

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamSingerTrajVar::Ptr>(time.nanosecs(), newEntry));
}

Eigen::Matrix<double,18,18> SteamSingerTrajInterface::precomputeInterpQinv(const double& dt) {
  double dt2=dt*dt;
  double dt3=dt2*dt;
  Eigen::Matrix<double,6,6> alpha2=alpha_*alpha_;
  Eigen::Matrix<double,6,6> alpha3=alpha2*alpha_;
  Eigen::Matrix<double,6,6> alpha2_inv=alpha_inv_*alpha_inv_;
  Eigen::Matrix<double,6,6> alpha3_inv=alpha2_inv*alpha_inv_;
  Eigen::Matrix<double,6,6> alpha4_inv=alpha3_inv*alpha_inv_;
  Eigen::Matrix<double,6,6> eye=Eigen::Matrix<double,6,6>::Identity();
  Eigen::Matrix<double,6,6> expon2; expon2.setZero();
  expon2.diagonal()=(-2*dt*alpha_).diagonal().array().exp();
  Eigen::Matrix<double,6,6> expon; expon.setZero();
  expon.diagonal()=(-dt*alpha_).diagonal().array().exp();
  
  Eigen::Matrix<double, 18, 18> Q_interp;
  Q_interp.block<6,6>(0,0) = alpha4_inv*(eye-expon2+2*alpha_*dt+(2.0/3.0)*alpha3*dt3-2*alpha2*dt2-4*alpha_*dt*expon);
  Q_interp.block<6,6>(6,6) = alpha2_inv*(4*expon-3*eye-expon2+2*alpha_*dt);
  Q_interp.block<6,6>(12,12) = (eye-expon2);;
  Q_interp.block<6,6>(6,0) = Q_interp.block<6,6>(0,6) = alpha3_inv*(expon2+eye-2*expon+2*alpha_*dt*expon-2*alpha_*dt+alpha2*dt2);
  Q_interp.block<6,6>(12,0) = Q_interp.block<6,6>(0,12) = alpha2_inv*(eye-expon2-2*alpha_*dt*expon);
  Q_interp.block<6,6>(12,6) = Q_interp.block<6,6>(6,12) = alpha_inv_*(expon2+eye-2*expon);

  Q_inv_interp_=Q_interp.inverse();
  return Q_inv_interp_;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
TransformEvaluator::ConstPtr SteamSingerTrajInterface::getInterpPoseEval(const steam::Time& time, bool usePrecomputedQinv) const {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

    // If we allow extrapolation
    if (allowExtrapolation_) {
      --it1; // should be safe, as we checked that the map was not empty..
      const SteamSingerTrajVar::Ptr& endKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          SingerTransformEvaluator::MakeShared(endKnot->getVelocity(),
                                                 endKnot->getAcceleration(),
                                                 time - endKnot->getTime(),
                                                 alpha_, alpha_inv_);
      return compose(T_t_k, endKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time. 1");
    }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {

    // return state variable exactly (no interp)
    return it1->second->getPose();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {

    if (allowExtrapolation_) {
      const SteamSingerTrajVar::Ptr& startKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
      SingerTransformEvaluator::MakeShared(startKnot->getVelocity(),
                                             startKnot->getAcceleration(),
                                             time - startKnot->getTime(),
                                             alpha_, alpha_inv_);
      return compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // Create interpolated evaluator
  if (usePrecomputedQinv) {
    return SteamSingerTrajPoseInterpEval::MakeShared(time, it1->second, it2->second, alpha_, alpha_inv_, Q_inv_interp_);
  }
  return SteamSingerTrajPoseInterpEval::MakeShared(time, it1->second, it2->second, alpha_, alpha_inv_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
///        should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajInterface::addPosePrior(const steam::Time& time,
                                      const lgmath::se3::Transformation& pose,
                                      const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] no knot at provided time.");
  }

  // Get reference
  const SteamSingerTrajVar::Ptr& knotRef = it->second;

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
void SteamSingerTrajInterface::addVelocityPrior(const steam::Time& time,
                                          const Eigen::Matrix<double,6,1>& velocity,
                                          const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const SteamSingerTrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(knotRef->getVelocity()->isLocked()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] tried to add prior to locked velocity.");
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
/// \brief Add a unary acceleration prior factor at a knot time. Note that only a single acceleration
///        prior should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajInterface::addAccelerationPrior(const steam::Time& time,
  const Eigen::Matrix<double,6,1>& acceleration,
  const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
  throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
  throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const SteamSingerTrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(knotRef->getAcceleration()->isLocked()) {
  throw std::runtime_error("[GpTrajectory][addAccelerationPrior] tried to add prior to locked acceleration.");
  }

  // Set up loss function, noise model, and error function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(cov));
  steam::VectorSpaceErrorEval<6,6>::Ptr errorfunc(new steam::VectorSpaceErrorEval<6,6>(acceleration, knotRef->getAcceleration()));

  // Create cost term
  accelerationPriorFactor_ = steam::WeightedLeastSqCostTerm<6,6>::Ptr(
    new steam::WeightedLeastSqCostTerm<6,6>(errorfunc, sharedNoiseModel, sharedLossFunc));
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get cost terms associated with the prior for unlocked parts of the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajInterface::appendPriorCostTerms(
    const ParallelizedCostTermCollection::Ptr& costTerms) const {

  // If empty, return none
  if (knotMap_.empty()) {
    return;
  }

  // Check for pose, velocity priors, and acceleration priors
  if (posePriorFactor_) {
    costTerms->add(posePriorFactor_);
  }
  if (velocityPriorFactor_) {
    costTerms->add(velocityPriorFactor_);
  }
  if (accelerationPriorFactor_) {
    costTerms->add(accelerationPriorFactor_);
  }
  // All prior factors will use an L2 loss function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  // steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1));

  // Initialize first iterator
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it1 = knotMap_.begin();
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("No knots...");
  }

  // Iterate through all states.. if any are unlocked, supply a prior term
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it2 = it1; ++it2;
  for (; it2 != knotMap_.end(); ++it1, ++it2) {

    // Get knots
    const SteamSingerTrajVar::ConstPtr& knot1 = it1->second;
    const SteamSingerTrajVar::ConstPtr& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if(knot1->getPose()->isActive()  || !knot1->getVelocity()->isLocked() ||
       knot2->getPose()->isActive()  || !knot2->getVelocity()->isLocked() ||
       !knot2->getAcceleration()->isLocked()  || !knot2->getAcceleration()->isLocked()) {

      // Generate 18 x 18 covarinace matrix for GP prior factor
      Eigen::Matrix<double,18,18> Qi;
      double dt=(knot2->getTime() - knot1->getTime()).seconds();
      // double one_over_dt = 1.0/(knot2->getTime() - knot1->getTime()).seconds();
      // double one_over_dt2 = one_over_dt*one_over_dt;
      // double one_over_dt3 = one_over_dt2*one_over_dt;
      // double one_over_dt4 = one_over_dt3*one_over_dt;
      // double one_over_dt5 = one_over_dt4*one_over_dt;

      Eigen::Matrix<double,6,6> eye=Eigen::Matrix<double,6,6>::Identity();
      Eigen::Matrix<double,6,6> expon2; expon2.setZero();
      expon2.diagonal()=(-2*dt*alpha_).diagonal().array().exp();
      Eigen::Matrix<double,6,6> expon; expon.setZero();
      expon.diagonal()=(-dt*alpha_).diagonal().array().exp();
      Eigen::Matrix<double,6,6> alpha2=alpha_*alpha_;
      Eigen::Matrix<double,6,6> alpha3=alpha2*alpha_;
      Eigen::Matrix<double,6,6> alpha2_inv=alpha_inv_*alpha_inv_;
      Eigen::Matrix<double,6,6> alpha3_inv=alpha2_inv*alpha_inv_;
      Eigen::Matrix<double,6,6> alpha4_inv=alpha3_inv*alpha_inv_;
      double dt2=dt*dt;
      double dt3=dt2*dt;

      Qi.block<6,6>(0,0) = Qc_*alpha4_inv*(eye-expon2+2*alpha_*dt+(2.0/3.0)*alpha3*dt3-2*alpha2*dt2-4*alpha_*dt*expon);
      Qi.block<6,6>(6,6) = Qc_*alpha2_inv*(4*expon-3*eye-expon2+2*alpha_*dt);
      Qi.block<6,6>(12,12) = Qc_*(eye-expon2);;
      Qi.block<6,6>(6,0) = Qi.block<6,6>(0,6) = Qc_*alpha3_inv*(expon2+eye-2*expon+2*alpha_*dt*expon-2*alpha_*dt+alpha2*dt2);
      Qi.block<6,6>(12,0) = Qi.block<6,6>(0,12) = Qc_*alpha2_inv*(eye-expon2-2*alpha_*dt*expon);
      Qi.block<6,6>(12,6) = Qi.block<6,6>(6,12) = Qc_*alpha_inv_*(expon2+eye-2*expon);
      
      // std::cout << Qi.block<6,6>(12,0) << std::endl << std::endl;
      // std::cout << Qi.block<6,6>(0,12) << std::endl;
      // std::cout << Qi << std::endl;
      steam::BaseNoiseModelX::Ptr sharedGPNoiseModel(
            new steam::StaticNoiseModelX(Qi/*, steam::INFORMATION*/));

      // Create cost term
      steam::se3::SteamSingerTrajPriorFactor::Ptr errorfunc(
            new steam::se3::SteamSingerTrajPriorFactor(knot1, knot2, alpha_, alpha_inv_));
      steam::WeightedLeastSqCostTermX::Ptr cost(
            new steam::WeightedLeastSqCostTermX(errorfunc, sharedGPNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajInterface::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  // Iterate over trajectory
  std::map<boost::int64_t, SteamSingerTrajVar::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {

    // Append active states in transform evaluator
    it->second->getPose()->getActiveStateVariables(outStates);

    // Check if velocity is locked
    if (!it->second->getVelocity()->isLocked()) {
      (*outStates)[it->second->getVelocity()->getKey().getID()] = it->second->getVelocity();
    }
  }
}

} // se3
} // steam
