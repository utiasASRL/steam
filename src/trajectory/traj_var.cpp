#include "steam/trajectory/traj_var.hpp"

namespace steam {
namespace traj {

TrajVar::TrajVar(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                 const Evaluable<VelocityType>::Ptr& w_0k_ink)
    : time_(time),
      T_k0_(T_k0),
      w_0k_ink_(w_0k_ink),
      cov_(CovType::Zero()),
      cov_set_(false) {}

TrajVar::TrajVar(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                 const Evaluable<VelocityType>::Ptr& w_0k_ink,
                 const CovType& cov)
    : time_(time),
      T_k0_(T_k0),
      w_0k_ink_(w_0k_ink),
      cov_(cov),
      cov_set_(true) {}

const Time& TrajVar::getTime() const { return time_; }

auto TrajVar::getPose() const -> const Evaluable<PoseType>::Ptr& {
  return T_k0_;
}

auto TrajVar::getVelocity() const -> const Evaluable<VelocityType>::Ptr& {
  return w_0k_ink_;
}

auto TrajVar::getCovariance() const -> const CovType& { return cov_; }

bool TrajVar::covarianceSet() const { return cov_set_; }

}  // namespace traj
}  // namespace steam
