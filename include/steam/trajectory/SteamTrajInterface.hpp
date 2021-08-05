//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajInterface.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_TRAJECTORY_INTERFACE_HPP
#define STEAM_TRAJECTORY_INTERFACE_HPP

#include <Eigen/Core>

#include <steam/common/Time.hpp>

#include <steam/trajectory/SteamTrajVar.hpp>

#include <steam/problem/WeightedLeastSqCostTerm.hpp>
#include <steam/problem/ParallelizedCostTermCollection.hpp>

#include <steam/solver/GaussNewtonSolverBase.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The trajectory class wraps a set of state variables to provide an interface
///        that allows for continuous-time pose interpolation.
//////////////////////////////////////////////////////////////////////////////////////////////
class SteamTrajInterface
{
 public:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  ///        Note that the weighting matrix, Qc, should be provided if prior factors are needed
  ///        for estimation. Without Qc the interpolation methods can be used safely.
  //////////////////////////////////////////////////////////////////////////////////////////////
  SteamTrajInterface(bool allowExtrapolation = false);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  SteamTrajInterface(const Eigen::Matrix<double,6,6>& Qc_inv, bool allowExtrapolation = false);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a new knot
  //////////////////////////////////////////////////////////////////////////////////////////////
  void add(const SteamTrajVar::Ptr& knot);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a new knot
  //////////////////////////////////////////////////////////////////////////////////////////////
  void add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity);

  void add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity,
           const Eigen::Matrix<double,12,12> cov);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a new knot
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity,
           const VectorSpaceStateVar::Ptr& acceleration);

  virtual void add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity,
           const VectorSpaceStateVar::Ptr& acceleration,
           const Eigen::Matrix<double,18,18> cov);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual TransformEvaluator::ConstPtr getInterpPoseEval(const steam::Time& time) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get velocity evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd getVelocity(const steam::Time& time);

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
  /// \brief Get binary cost terms associated with the prior for active parts of the trajectory
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendPriorCostTerms(const ParallelizedCostTermCollection::Ptr& costTerms) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get active state variables in the trajectory
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void getActiveStateVariables(
      std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get pose prior cost
  //////////////////////////////////////////////////////////////////////////////////////////////
  double getPosePriorCost();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get velocity prior cost
  //////////////////////////////////////////////////////////////////////////////////////////////
  double getVelocityPriorCost();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Store solver in trajectory, needed for querying covariances later
  //////////////////////////////////////////////////////////////////////////////////////////////
  void setSolver(std::shared_ptr<GaussNewtonSolverBase> solver);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get interpolated/extrapolated covariance at given time
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd getCovariance(const steam::Time& time) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get interpolated/extrapolated covariance on a relative pose between t_a and t_b
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd getRelativeCovariance(const steam::Time& time_a, const steam::Time& time_b) const;

 protected:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Covariance translation from global(2x2 block matrix, each 6x6) to local
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd translateCovToLocal(const steam::BlockMatrix& global_cov,
      const SteamTrajVar::ConstPtr& knot1, const SteamTrajVar::ConstPtr& knot2) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Covariance translation from global(6x6 matrix, meaning 1 state) to local
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd translateCovToLocal(const Eigen::MatrixXd& global_cov,
      const SteamTrajVar::ConstPtr& knot1) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Covariance translation of interpolated covariance from local to global
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd translateCovToGlobal(const Eigen::MatrixXd& local_cov,
    const Eigen::MatrixXd& global_frame_cov, const lgmath::se3::Transformation& local_pose,
    const Eigen::MatrixXd& velocity) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Covariance translation of off-diagonal interpolated covariance from local to global
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd translateCrossCovToGlobal(const Eigen::MatrixXd& local_cov_ab,
    const Eigen::MatrixXd& global_frame_cov, const lgmath::se3::Transformation& local_pose_a,
    const Eigen::MatrixXd& velocity_a, const lgmath::se3::Transformation& local_pose_b,
    const Eigen::MatrixXd& velocity_b) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get interpolated state at given time
  //////////////////////////////////////////////////////////////////////////////////////////////
  void interpState(lgmath::se3::Transformation* pose, Eigen::MatrixXd* velocity,
      const Eigen::MatrixXd& lambda_psi, const steam::Time& time,
      const SteamTrajVar::ConstPtr& knot1, const SteamTrajVar::ConstPtr& knot2) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Compute covariance interpolation at given time
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd interpCovariance(const steam::Time& time) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Compute covariance interpolation at given time
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::MatrixXd extrapCovariance(const steam::Time& time) const;

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
  /// \brief Ordered map of knots
  //////////////////////////////////////////////////////////////////////////////////////////////
  std::map<boost::int64_t, SteamTrajVar::Ptr> knotMap_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Solver used, we use this to query covariances
  //////////////////////////////////////////////////////////////////////////////////////////////
  std::shared_ptr<GaussNewtonSolverBase> solver_;

};

} // se3
} // steam

#endif // STEAM_TRAJECTORY_INTERFACE_HPP
