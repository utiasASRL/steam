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

  static Ptr MakeShared(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                        const Evaluable<VelocityType>::Ptr& w_0k_ink) {
    return std::make_shared<Variable>(time, T_k0, w_0k_ink);
  }

  /** \brief Constructor */
  Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
           const Evaluable<VelocityType>::Ptr& w_0k_ink)
      : time_(time), T_k0_(T_k0), w_0k_ink_(w_0k_ink) {}

  /** \brief Constructor \todo should not be a base class */
  virtual ~Variable() = default;

  /** \brief Get timestamp */
  const Time& getTime() const { return time_; }

  /** \brief Get pose evaluator */
  const Evaluable<PoseType>::Ptr& getPose() const { return T_k0_; }

  /** \brief Get velocity state variable */
  const Evaluable<VelocityType>::Ptr& getVelocity() const { return w_0k_ink_; }

 private:
  /** \brief Timestamp of trajectory variable */
  Time time_;

  /** \brief Pose evaluator */
  const Evaluable<PoseType>::Ptr T_k0_;

  /** \brief Generalized 6D velocity state variable */
  const Evaluable<VelocityType>::Ptr w_0k_ink_;
};

}  // namespace const_vel
}  // namespace traj
}  // namespace steam