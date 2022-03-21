#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/traj_time.hpp"

namespace steam {
namespace traj {

/**
 * \brief This class wraps a pose and velocity evaluator to act as a
 * discrete-time trajectory state variable for continuous-time trajectory
 * estimation.
 */
class TrajVar {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<TrajVar>;
  using ConstPtr = std::shared_ptr<const TrajVar>;

  using PoseType = lgmath::se3::Transformation;
  using VelocityType = Eigen::Matrix<double, 6, 1>;
  using CovType = Eigen::Matrix<double, 12, 12>;

  /** \brief Constructor */
  TrajVar(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
          const Evaluable<VelocityType>::Ptr& w_0k_ink);
  TrajVar(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
          const Evaluable<VelocityType>::Ptr& w_0k_ink, const CovType& cov);

  /** \brief Constructor \todo should not be a base class */
  virtual ~TrajVar() = default;

  /** \brief Get timestamp */
  const Time& getTime() const;

  /** \brief Get pose evaluator */
  const Evaluable<PoseType>::Ptr& getPose() const;

  /** \brief Get velocity state variable */
  const Evaluable<VelocityType>::Ptr& getVelocity() const;

  const CovType& getCovariance() const;

  bool covarianceSet() const;

 private:
  /** \brief Timestamp of trajectory variable */
  Time time_;

  /** \brief Pose evaluator */
  const Evaluable<PoseType>::Ptr T_k0_;

  /** \brief Generalized 6D velocity state variable */
  const Evaluable<VelocityType>::Ptr w_0k_ink_;

  /** \brief Covariance */
  CovType cov_;
  bool cov_set_ = false;
};

}  // namespace traj
}  // namespace steam
