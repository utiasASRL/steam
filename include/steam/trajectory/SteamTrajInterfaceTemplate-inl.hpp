//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajInterfaceTemplate.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/SteamTrajInterfaceTemplate.hpp>

#include <lgmath.hpp>

#include <steam/trajectory/SteamTrajPoseInterpEval.hpp>
#include <steam/trajectory/SteamTrajPriorFactor.hpp>
#include <steam/evaluator/samples/VectorSpaceErrorEval.hpp>

#include <steam/evaluator/blockauto/transform/TransformEvalOperations.hpp>
#include <steam/evaluator/blockauto/transform/ConstVelTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename STV>
void SteamTrajInterfaceTemplate<STV>::add(const typename STV::Ptr& knot) {

  // Todo, check that time does not already exist in map?

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, typename STV::Ptr>(knot->getTime().nanosecs(), knot));
}

// //////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
///        should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename STV>
void SteamTrajInterfaceTemplate<STV>::addPosePrior(const steam::Time& time,
                                      const lgmath::se3::Transformation& pose,
                                      const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] map was empty.");
  }

  // Try to find knot at same time
  typename std::map<boost::int64_t, typename STV::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] no knot at provided time.");
  }

  // Get reference
  const typename STV::Ptr& knotRef = it->second;

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
template<typename STV>
void SteamTrajInterfaceTemplate<STV>::addVelocityPrior(const steam::Time& time,
                                          const Eigen::Matrix<double,6,1>& velocity,
                                          const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  typename std::map<boost::int64_t, typename STV::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const typename STV::Ptr& knotRef = it->second;

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
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename STV>
void SteamTrajInterfaceTemplate<STV>::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  // Iterate over trajectory
  typename std::map<boost::int64_t, typename STV::Ptr>::const_iterator it;
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
