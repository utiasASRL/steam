#include "steam/trajectory/const_vel/variable.hpp"
#include "steam/evaluable/se3/se3_state_var.hpp"
#include "steam/evaluable/vspace/vspace_state_var.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto Variable::MakeShared(const Time& time,
                          const Evaluable<PoseType>::Ptr& T_k0,
                          const Evaluable<VelocityType>::Ptr& w_0k_ink) -> Ptr {
  return std::make_shared<Variable>(time, T_k0, w_0k_ink);
}

// auto Variable::MakeShared(const Time& time,
//                           const Evaluable<PoseType>::Ptr& T_k0,
//                           const Evaluable<VelocityType>::Ptr& w_0k_ink,
//                           const CovType& cov) -> Ptr {
//   return std::make_shared<Variable>(time, T_k0, w_0k_ink, cov);
// }

Variable::Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                   const Evaluable<VelocityType>::Ptr& w_0k_ink)
    : time_(time),
      T_k0_(T_k0),
      w_0k_ink_(w_0k_ink),
      // cov_(CovType::Zero()),
      cov_set_(false) {}

// Variable::Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
//                    const Evaluable<VelocityType>::Ptr& w_0k_ink,
//                    const CovType& cov)
//     : time_(time),
//       T_k0_(T_k0),
//       w_0k_ink_(w_0k_ink),
//       cov_(cov),
//       cov_set_(true) {}

const Time& Variable::getTime() const { return time_; }

auto Variable::getPose() const -> const Evaluable<PoseType>::Ptr& {
  return T_k0_;
}

auto Variable::getVelocity() const -> const Evaluable<VelocityType>::Ptr& {
  return w_0k_ink_;
}

auto Variable::getActiveKeys() const -> std::vector<StateKey> {
  std::vector<StateKey> output;
  if (getPose()->active())
    output.push_back(dynamic_cast<se3::SE3StateVar&>(*getPose()).key());
  if (getVelocity()->active())
    output.push_back(dynamic_cast<vspace::VSpaceStateVar<6>&>(*getVelocity()).key());
  return output;
}

auto Variable::getCovariance() const -> const CovType& {
  if (!cov_set_)
    throw std::runtime_error(
        "[ConstVelTraj][Variable] requested covariance without setting.");
  return cov_; 
}

auto Variable::getCrossCov() const -> const CovType& {
  if (!cross_cov_set_)
    throw std::runtime_error(
        "[ConstVelTraj][Variable] requested cross-cov. without setting.");
  return cross_cov_;
}

bool Variable::covarianceSet() const { return cov_set_; }

bool Variable::crossCovSet() const { return cross_cov_set_; }

void Variable::setCovariance(const CovType& cov) {
  cov_ = cov;
  cov_set_ = true;
}

void Variable::setCrossCov(const CovType& cov) {
  cross_cov_ = cov;
  cross_cov_set_ = true;
}

void Variable::resetCovariance() {
  cov_set_ = false;
  cross_cov_set_ = false;
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam
