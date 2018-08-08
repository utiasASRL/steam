//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamCATrajInterface.hpp
///
/// \author Tim Tang, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_CA_TRAJECTORY_INTERFACE_HPP
#define STEAM_CA_TRAJECTORY_INTERFACE_HPP

#include <Eigen/Core>

#include <steam/common/Time.hpp>

#include <steam/trajectory_ca/SteamCATrajVar.hpp>

#include <steam/problem/WeightedLeastSqCostTerm.hpp>
#include <steam/problem/ParallelizedCostTermCollection.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The trajectory class wraps a set of state variables to provide an interface
///        that allows for continuous-time pose interpolation.
//////////////////////////////////////////////////////////////////////////////////////////////
class SteamCATrajInterface
{
 public:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  ///        Note that the weighting matrix, Qc, should be provided if prior factors are needed
  ///        for estimation. Without Qc the interpolation methods can be used safely.
  //////////////////////////////////////////////////////////////////////////////////////////////
  SteamCATrajInterface(bool allowExtrapolation = false);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  SteamCATrajInterface(const Eigen::Matrix<double,6,6>& Qc_inv, bool allowExtrapolation = false);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a new knot
  //////////////////////////////////////////////////////////////////////////////////////////////
  void add(const SteamCATrajVar::Ptr& knot);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a new knot
  //////////////////////////////////////////////////////////////////////////////////////////////
  void add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity,
           const VectorSpaceStateVar::Ptr& acceleration);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  TransformEvaluator::ConstPtr getInterpPoseEval(const steam::Time& time) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
  ///        should exist on a trajectory, adding a second will overwrite the first.
  //////////////////////////////////////////////////////////////////////////////////////////////
  void addPosePrior(const steam::Time& time, const lgmath::se3::Transformation& pose,
                    const Eigen::Matrix<double,6,6>& cov);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a unary velocity prior factor at a knot time. Note that only a single velocity
  ///        prior should exist on a trajectory, adding a second will overwrite the first.
  //////////////////////////////////////////////////////////////////////////////////////////////
  void addVelocityPrior(const steam::Time& time, const Eigen::Matrix<double,6,1>& velocity,
                        const Eigen::Matrix<double,6,6>& cov);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a unary acceleration prior factor at a knot time. Note that only a single acceleration
  ///        prior should exist on a trajectory, adding a second will overwrite the first.
  //////////////////////////////////////////////////////////////////////////////////////////////
  void addAccelerationPrior(const steam::Time& time, const Eigen::Matrix<double,6,1>& acceleration,
    const Eigen::Matrix<double,6,6>& cov);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get binary cost terms associated with the prior for active parts of the trajectory
  //////////////////////////////////////////////////////////////////////////////////////////////
  void appendPriorCostTerms(const ParallelizedCostTermCollection::Ptr& costTerms) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get active state variables in the trajectory
  //////////////////////////////////////////////////////////////////////////////////////////////
  void getActiveStateVariables(
      std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const;

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Ordered map of knots
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double,6,6> Qc_inv_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Allow for extrapolation
  //////////////////////////////////////////////////////////////////////////////////////////////
  bool allowExtrapolation_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pose prior
  //////////////////////////////////////////////////////////////////////////////////////////////
  steam::WeightedLeastSqCostTerm<6,6>::Ptr posePriorFactor_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Velocity prior
  //////////////////////////////////////////////////////////////////////////////////////////////
  steam::WeightedLeastSqCostTerm<6,6>::Ptr velocityPriorFactor_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Acceleration prior
  //////////////////////////////////////////////////////////////////////////////////////////////
  steam::WeightedLeastSqCostTerm<6,6>::Ptr accelerationPriorFactor_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Ordered map of knots
  //////////////////////////////////////////////////////////////////////////////////////////////
  std::map<boost::int64_t, SteamCATrajVar::Ptr> knotMap_;

};

} // se3
} // steam

#endif // STEAM_CA_TRAJECTORY_INTERFACE_HPP
