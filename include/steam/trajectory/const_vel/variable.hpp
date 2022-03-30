#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_vel {

/**
 * \brief This class wraps a pose and velocity evaluator to act as a
 * discrete-time trajectory state variable for continuous-time trajectory
 * estimation.
 */
class Variable {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<Variable>;
  using ConstPtr = std::shared_ptr<const Variable>;

  using PoseType = lgmath::se3::Transformation;
  using VelocityType = Eigen::Matrix<double, 6, 1>;
  using CovType = Eigen::Matrix<double, 12, 12>;

  static Ptr MakeShared(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                        const Evaluable<VelocityType>::Ptr& w_0k_ink);
  static Ptr MakeShared(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                        const Evaluable<VelocityType>::Ptr& w_0k_ink,
                        const CovType& cov);

  /** \brief Constructor */
  Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
           const Evaluable<VelocityType>::Ptr& w_0k_ink);
  Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
           const Evaluable<VelocityType>::Ptr& w_0k_ink, const CovType& cov);

  /** \brief Constructor \todo should not be a base class */
  virtual ~Variable() = default;

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

}  // namespace const_vel
}  // namespace traj
}  // namespace steam