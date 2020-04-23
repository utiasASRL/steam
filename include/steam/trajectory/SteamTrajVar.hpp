//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajVar.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_TRAJECTORY_VAR_HPP
#define STEAM_TRAJECTORY_VAR_HPP

#include <Eigen/Core>

#include <steam/common/Time.hpp>
#include <steam/evaluator/blockauto/transform/TransformEvaluator.hpp>
#include <steam/state/VectorSpaceStateVar.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief This class wraps a pose and velocity evaluator to act as a discrete-time trajectory
///        state variable for continuous-time trajectory estimation.
//////////////////////////////////////////////////////////////////////////////////////////////
class SteamTrajVar
{
 public:

  /// Shared pointer typedefs for readability
  typedef boost::shared_ptr<SteamTrajVar> Ptr;
  typedef boost::shared_ptr<const SteamTrajVar> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  SteamTrajVar(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
               const VectorSpaceStateVar::Ptr& velocity);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get pose evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  const se3::TransformEvaluator::Ptr& getPose() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get velocity state variable
  //////////////////////////////////////////////////////////////////////////////////////////////
  const VectorSpaceStateVar::Ptr& getVelocity() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get acceleration state variable, to be overriden
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual const VectorSpaceStateVar::Ptr& getAcceleration() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get timestamp
  //////////////////////////////////////////////////////////////////////////////////////////////
  const steam::Time& getTime() const;

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Timestamp of trajectory variable
  //////////////////////////////////////////////////////////////////////////////////////////////
  steam::Time time_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pose evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  se3::TransformEvaluator::Ptr T_k0_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Generalized 6D velocity state variable
  //////////////////////////////////////////////////////////////////////////////////////////////
  VectorSpaceStateVar::Ptr velocity_;
};

} // se3
} // steam

#endif // STEAM_TRAJECTORY_VAR_HPP
