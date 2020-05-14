//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajVar.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/SteamTrajVar.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamTrajVar::SteamTrajVar(const steam::Time& time,
             const se3::TransformEvaluator::Ptr& T_k0,
             const VectorSpaceStateVar::Ptr& velocity)
  : time_(time), T_k0_(T_k0), velocity_(velocity) {

  // Check velocity input
  if (velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get pose evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
const se3::TransformEvaluator::Ptr& SteamTrajVar::getPose() const {
  return T_k0_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get velocity state variable
//////////////////////////////////////////////////////////////////////////////////////////////
const VectorSpaceStateVar::Ptr& SteamTrajVar::getVelocity() const {
  return velocity_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get acceleration state variable, to be overriden
//////////////////////////////////////////////////////////////////////////////////////////////
const VectorSpaceStateVar::Ptr& SteamTrajVar::getAcceleration() const {
  throw std::runtime_error("Steam trajectory variable does not have an acceleration state!");
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get timestamp
//////////////////////////////////////////////////////////////////////////////////////////////
const steam::Time& SteamTrajVar::getTime() const {
  return time_;
}

} // se3
} // steam
