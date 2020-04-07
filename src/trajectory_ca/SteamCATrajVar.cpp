//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamCATrajVar.cpp
///
/// \author Tim Tang, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory_ca/SteamCATrajVar.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamCATrajVar::SteamCATrajVar(const steam::Time& time,
             const se3::TransformEvaluator::Ptr& T_k0,
             const VectorSpaceStateVar::Ptr& velocity,
             const VectorSpaceStateVar::Ptr& acceleration)
  : SteamTrajVar(time, T_k0, velocity), acceleration_(acceleration) {

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
/// \brief Get acceleration state variable
//////////////////////////////////////////////////////////////////////////////////////////////
const VectorSpaceStateVar::Ptr& SteamCATrajVar::getAcceleration() const {
  return acceleration_;
}

} // se3
} // steam
