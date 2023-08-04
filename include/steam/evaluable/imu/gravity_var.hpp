#pragma once

#include "lgmath.hpp"

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace imu {

class GravityStateVar : public StateVar<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<GravityStateVar>;
  using ConstPtr = std::shared_ptr<const GravityStateVar>;

  using T = Eigen::Matrix<double, 3, 1>;
  using Base = StateVar<T>;

  static Ptr MakeShared(const T& value, const std::string& name = "");
  GravityStateVar(const T& value, const std::string& name = "");

  bool update(const Eigen::VectorXd& perturbation) override;
  StateVarBase::Ptr clone() const override;
};

auto GravityStateVar::MakeShared(const T& value, const std::string& name)
    -> Ptr {
  return std::make_shared<GravityStateVar>(value, name);
}

// Note: perturbation dimension is 2
GravityStateVar::GravityStateVar(const T& value, const std::string& name)
    : Base(value, 2, name) {}

bool GravityStateVar::update(const Eigen::VectorXd& perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error(
        "GravityStateVar::update: perturbation size mismatch");
  Eigen::Vector3d rot_perturb;
  rot_perturb << perturbation(0, 0), perturbation(1, 0), 0;
  value_ = lgmath::so3::Rotation(rot_perturb) * value_;
  return true;
}

StateVarBase::Ptr GravityStateVar::clone() const {
  return std::make_shared<GravityStateVar>(*this);
}

}  // namespace imu
}  // namespace steam