//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamSingerTrajVar.cpp
///
/// \author Jeremy Wong, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory_singer/SteamSingerTrajVar.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamSingerTrajVar::SteamSingerTrajVar(const steam::Time& time,
             const se3::TransformEvaluator::Ptr& T_k0,
             const VectorSpaceStateVar::Ptr& velocity,
             const VectorSpaceStateVar::Ptr& acceleration)
  : time_(time), T_k0_(T_k0), velocity_(velocity), acceleration_(acceleration) {

  // Check velocity input
  if (velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }

  // Check acceleration input
  if (acceleration->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid acceleration size");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get pose evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
const se3::TransformEvaluator::Ptr& SteamSingerTrajVar::getPose() const {
  return T_k0_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get velocity state variable
//////////////////////////////////////////////////////////////////////////////////////////////
const VectorSpaceStateVar::Ptr& SteamSingerTrajVar::getVelocity() const {
  return velocity_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get acceleration state variable
//////////////////////////////////////////////////////////////////////////////////////////////
const VectorSpaceStateVar::Ptr& SteamSingerTrajVar::getAcceleration() const {
  return acceleration_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get timestamp
//////////////////////////////////////////////////////////////////////////////////////////////
const steam::Time& SteamSingerTrajVar::getTime() const {
  return time_;
}

} // se3
} // steam
