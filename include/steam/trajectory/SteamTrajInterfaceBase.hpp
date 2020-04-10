//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajInterface.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_TRAJECTORY_INTERFACE_BASE_HPP
#define STEAM_TRAJECTORY_INTERFACE_BASE_HPP

#include <Eigen/Core>

#include <steam/common/Time.hpp>

#include <steam/problem/WeightedLeastSqCostTerm.hpp>
#include <steam/problem/ParallelizedCostTermCollection.hpp>

#include <steam/evaluator/blockauto/transform/TransformEvalOperations.hpp>
#include <steam/evaluator/blockauto/transform/ConstVelTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The trajectory class wraps a set of state variables to provide an interface
///        that allows for continuous-time pose interpolation.
//////////////////////////////////////////////////////////////////////////////////////////////
class SteamTrajInterfaceBase
{
 public:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  ///        Note that the weighting matrix, Qc, should be provided if prior factors are needed
  ///        for estimation. Without Qc the interpolation methods can be used safely.
  //////////////////////////////////////////////////////////////////////////////////////////////
  SteamTrajInterfaceBase(bool allowExtrapolation = false);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  SteamTrajInterfaceBase(const Eigen::Matrix<double,6,6>& Qc_inv, bool allowExtrapolation = false);

  double getPosePriorCost();
  double getVelocityPriorCost();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a new knot, to be implemented
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void add(const steam::Time& time,
                   const se3::TransformEvaluator::Ptr& T_k0,
                   const VectorSpaceStateVar::Ptr& velocity);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a new knot, to be overriden if acceleration exists
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
             const VectorSpaceStateVar::Ptr& velocity,
             const VectorSpaceStateVar::Ptr& acceleration);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual TransformEvaluator::ConstPtr getInterpPoseEval(const steam::Time& time) const = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get velocity evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd getVelocity(const steam::Time& time) = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
  ///        should exist on a trajectory, adding a second will overwrite the first.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void addPosePrior(const steam::Time& time, const lgmath::se3::Transformation& pose,
                    const Eigen::Matrix<double,6,6>& cov) = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a unary velocity prior factor at a knot time. Note that only a single velocity
  ///        prior should exist on a trajectory, adding a second will overwrite the first.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void addVelocityPrior(const steam::Time& time, const Eigen::Matrix<double,6,1>& velocity,
                        const Eigen::Matrix<double,6,6>& cov) = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get binary cost terms associated with the prior for active parts of the trajectory
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendPriorCostTerms(const ParallelizedCostTermCollection::Ptr& costTerms) const = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get active state variables in the trajectory
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void getActiveStateVariables(
      std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const = 0;


 protected:

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

};

} // se3
} // steam

#endif // STEAM_TRAJECTORY_INTERFACE_BASE_HPP
