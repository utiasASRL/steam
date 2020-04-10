//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajInterface.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/SteamTrajInterfaceBase.hpp>

namespace steam {
namespace se3 {

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
///        Note that the weighting matrix, Qc, should be provided if prior factors are needed
///        for estimation. Without Qc the interpolation methods can be used safely.
//////////////////////////////////////////////////////////////////////////////////////////////
SteamTrajInterfaceBase::SteamTrajInterfaceBase(bool allowExtrapolation) :
  Qc_inv_(Eigen::Matrix<double,6,6>::Identity()), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamTrajInterfaceBase::SteamTrajInterfaceBase(const Eigen::Matrix<double,6,6>& Qc_inv, bool allowExtrapolation) :
  Qc_inv_(Qc_inv), allowExtrapolation_(allowExtrapolation) {
}

double SteamTrajInterfaceBase::getPosePriorCost() {
  if(posePriorFactor_ != nullptr) {
    return posePriorFactor_->cost();
  } else {
    return 0.0;
  }
}

double SteamTrajInterfaceBase::getVelocityPriorCost() {
  if(velocityPriorFactor_ != nullptr) {
    return velocityPriorFactor_->cost();
  } else {
    return 0.0;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot, to be implemented
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterfaceBase::add(const steam::Time& time,
                             const se3::TransformEvaluator::Ptr& T_k0,
                             const VectorSpaceStateVar::Ptr& velocity) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot, to be overriden if acceleration exists
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterfaceBase::add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity,
           const VectorSpaceStateVar::Ptr& acceleration) {
  add(time, T_k0, velocity);
}

} // se3
} // steam
