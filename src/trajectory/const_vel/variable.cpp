#include "steam/trajectory/const_vel/variable.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto Variable::MakeShared(const Time& time,
                          const Evaluable<PoseType>::Ptr& T_k0,
                          const Evaluable<VelocityType>::Ptr& w_0k_ink) -> Ptr {
  return std::make_shared<Variable>(time, T_k0, w_0k_ink);
}

auto Variable::MakeShared(const Time& time,
                          const Evaluable<PoseType>::Ptr& T_k0,
                          const Evaluable<VelocityType>::Ptr& w_0k_ink,
                          const CovType& cov) -> Ptr {
  return std::make_shared<Variable>(time, T_k0, w_0k_ink, cov);
}

Variable::Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                   const Evaluable<VelocityType>::Ptr& w_0k_ink)
    : time_(time),
      T_k0_(T_k0),
      w_0k_ink_(w_0k_ink),
      cov_(CovType::Zero()),
      cov_set_(false) {}

Variable::Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                   const Evaluable<VelocityType>::Ptr& w_0k_ink,
                   const CovType& cov)
    : time_(time),
      T_k0_(T_k0),
      w_0k_ink_(w_0k_ink),
      cov_(cov),
      cov_set_(true) {}

const Time& Variable::getTime() const { return time_; }

auto Variable::getPose() const -> const Evaluable<PoseType>::Ptr& {
  return T_k0_;
}

auto Variable::getVelocity() const -> const Evaluable<VelocityType>::Ptr& {
  return w_0k_ink_;
}

auto Variable::getCovariance() const -> const CovType& { return cov_; }

bool Variable::covarianceSet() const { return cov_set_; }

}  // namespace const_vel
}  // namespace traj
}  // namespace steam
