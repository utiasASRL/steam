//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamCATrajInterface.cpp
///
/// \author Tim Tang, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory_ca/SteamCATrajInterface.hpp>

#include <lgmath.hpp>

#include <steam/trajectory_ca/SteamCATrajPoseInterpEval.hpp>
#include <steam/trajectory_ca/SteamCATrajPriorFactor.hpp>
#include <steam/evaluator/samples/VectorSpaceErrorEval.hpp>

#include <steam/evaluator/blockauto/transform/TransformEvalOperations.hpp>
#include <steam/evaluator/blockauto/transform/ConstVelTransformEvaluator.hpp>
#include <steam/evaluator/blockauto/transform/ConstAccTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
///        Note, without providing Qc, the trajectory can be used safely for interpolation,
///        but should not be used for estimation.
//////////////////////////////////////////////////////////////////////////////////////////////
SteamCATrajInterface::SteamCATrajInterface(bool allowExtrapolation) :
  Qc_inv_(Eigen::Matrix<double,6,6>::Identity()), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamCATrajInterface::SteamCATrajInterface(const Eigen::Matrix<double,6,6>& Qc_inv,
                                       bool allowExtrapolation) :
  Qc_inv_(Qc_inv), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::add(const SteamCATrajVar::Ptr& knot) {

  // Todo, check that time does not already exist in map?

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamCATrajVar::Ptr>(knot->getTime().nanosecs(), knot));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::add(const steam::Time& time,
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
  SteamCATrajVar::Ptr newEntry(new SteamCATrajVar(time, T_k0, velocity, acceleration));

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamCATrajVar::Ptr>(time.nanosecs(), newEntry));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
TransformEvaluator::ConstPtr SteamCATrajInterface::getInterpPoseEval(const steam::Time& time) const {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      --it1; // should be safe, as we checked that the map was not empty..
      const SteamCATrajVar::Ptr& endKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstAccTransformEvaluator::MakeShared(endKnot->getVelocity(),
                                                 endKnot->getAcceleration(),
                                                 time - endKnot->getTime());
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
      const SteamCATrajVar::Ptr& startKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(startKnot->getVelocity(),
                                                 time - startKnot->getTime());
      return compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // Create interpolated evaluator
  return SteamCATrajPoseInterpEval::MakeShared(time, it1->second, it2->second);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
///        should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::addPosePrior(const steam::Time& time,
                                      const lgmath::se3::Transformation& pose,
                                      const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] no knot at provided time.");
  }

  // Get reference
  const SteamCATrajVar::Ptr& knotRef = it->second;

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
void SteamCATrajInterface::addVelocityPrior(const steam::Time& time,
                                          const Eigen::Matrix<double,6,1>& velocity,
                                          const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const SteamCATrajVar::Ptr& knotRef = it->second;

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
void SteamCATrajInterface::appendPriorCostTerms(
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
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it1 = knotMap_.begin();
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("No knots...");
  }

  // Iterate through all states.. if any are unlocked, supply a prior term
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it2 = it1; ++it2;
  for (; it2 != knotMap_.end(); ++it1, ++it2) {

    // Get knots
    const SteamCATrajVar::ConstPtr& knot1 = it1->second;
    const SteamCATrajVar::ConstPtr& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if(knot1->getPose()->isActive()  || !knot1->getVelocity()->isLocked() ||
       knot2->getPose()->isActive()  || !knot2->getVelocity()->isLocked() ) {

      // Generate 18 x 18 information matrix for GP prior factor
      Eigen::Matrix<double,18,18> Qi_inv;
      double one_over_dt = 1.0/(knot2->getTime() - knot1->getTime()).seconds();
      double one_over_dt2 = one_over_dt*one_over_dt;
      double one_over_dt3 = one_over_dt2*one_over_dt;
      double one_over_dt4 = one_over_dt3*one_over_dt;
      double one_over_dt5 = one_over_dt4*one_over_dt;

      Qi_inv.block<6,6>(0,0) = 720.0 * one_over_dt5 * Qc_inv_;
      Qi_inv.block<6,6>(6,6) = 192.0 * one_over_dt3  * Qc_inv_;
      Qi_inv.block<6,6>(12,12) = 9.0 * one_over_dt  * Qc_inv_;
      Qi_inv.block<6,6>(6,0) = Qi_inv.block<6,6>(0,6) = -360.0 * one_over_dt4 * Qc_inv_;
      Qi_inv.block<6,6>(12,0) = Qi_inv.block<6,6>(0,12) = 60.0 * one_over_dt3 * Qc_inv_;
      Qi_inv.block<6,6>(12,6) = Qi_inv.block<6,6>(6,12) = -36.0 * one_over_dt2 * Qc_inv_;
      
      steam::BaseNoiseModelX::Ptr sharedGPNoiseModel(
            new steam::StaticNoiseModelX(Qi_inv, steam::INFORMATION));

      // Create cost term
      steam::se3::SteamCATrajPriorFactor::Ptr errorfunc(
            new steam::se3::SteamCATrajPriorFactor(knot1, knot2));
      steam::WeightedLeastSqCostTermX::Ptr cost(
            new steam::WeightedLeastSqCostTermX(errorfunc, sharedGPNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  // Iterate over trajectory
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it;
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
