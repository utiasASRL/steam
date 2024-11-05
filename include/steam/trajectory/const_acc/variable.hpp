#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_acc {

class Variable {
 public:
  using Ptr = std::shared_ptr<Variable>;
  using ConstPtr = std::shared_ptr<const Variable>;

  using PoseType = lgmath::se3::Transformation;
  using VelocityType = Eigen::Matrix<double, 6, 1>;
  using AccelerationType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Time time, const Evaluable<PoseType>::Ptr& T_k0,
                        const Evaluable<VelocityType>::Ptr& w_0k_ink,
                        const Evaluable<AccelerationType>::Ptr& dw_0k_ink) {
    return std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink);
  }

  Variable(const Time time, const Evaluable<PoseType>::Ptr& T_k0,
           const Evaluable<VelocityType>::Ptr& w_0k_ink,
           const Evaluable<AccelerationType>::Ptr& dw_0k_ink)
      : time_(time), T_k0_(T_k0), w_0k_ink_(w_0k_ink), dw_0k_ink_(dw_0k_ink) {}

  virtual ~Variable() = default;

  // clang-format off
  const Time& time() const { return time_; }
  const Evaluable<PoseType>::Ptr& pose() const { return T_k0_; }
  const Evaluable<VelocityType>::Ptr& velocity() const { return w_0k_ink_; }
  const Evaluable<AccelerationType>::Ptr& acceleration() const { return dw_0k_ink_; }
  // clang-format on

 private:
  Time time_;
  const Evaluable<PoseType>::Ptr T_k0_;
  const Evaluable<VelocityType>::Ptr w_0k_ink_;
  const Evaluable<AccelerationType>::Ptr dw_0k_ink_;
};

}  // namespace const_acc
}  // namespace traj
}  // namespace steam