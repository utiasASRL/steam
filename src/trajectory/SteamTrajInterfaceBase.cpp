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

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get transform evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
TransformEvaluator::ConstPtr SteamTrajInterfaceBase::getInterpPoseEval(const steam::Time& time) const {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get velocity evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd SteamTrajInterfaceBase::getVelocity(const steam::Time& time) {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
///        should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterfaceBase::addPosePrior(const steam::Time& time, const lgmath::se3::Transformation& pose,
                  const Eigen::Matrix<double,6,6>& cov) {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary velocity prior factor at a knot time. Note that only a single velocity
///        prior should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterfaceBase::addVelocityPrior(const steam::Time& time, const Eigen::Matrix<double,6,1>& velocity,
                      const Eigen::Matrix<double,6,6>& cov) {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary acceleration prior factor at a knot time. Note that only a single acceleration
///        prior should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterfaceBase::addAccelerationPrior(const steam::Time& time, const Eigen::Matrix<double,6,1>& acceleration,
  const Eigen::Matrix<double,6,6>& cov) {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get binary cost terms associated with the prior for active parts of the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterfaceBase::appendPriorCostTerms(const ParallelizedCostTermCollection::Ptr& costTerms) const {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterfaceBase::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {}

} // se3
} // steam
