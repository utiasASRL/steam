#pragma once

#include "lgmath.hpp"

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace vspace {

template <int DIM = Eigen::Dynamic>
class PreIntVelocityStateVar : public StateVar<Eigen::Matrix<double, DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<PreIntVelocityStateVar>;
  using ConstPtr = std::shared_ptr<const PreIntVelocityStateVar>;

  using T = Eigen::Matrix<double, DIM, 1>;
  using Base = StateVar<T>;
  using InType = lgmath::se3::Transformation;

  static Ptr MakeShared(const T& value, const Evaluable<InType>::ConstPtr& T_iv, const std::string& name = "");
  PreIntVelocityStateVar(const T& value, const Evaluable<InType>::ConstPtr& T_iv, const std::string& name = "");

  bool update(const Eigen::VectorXd& perturbation) override;
  StateVarBase::Ptr clone() const override;\

 private:
  const Evaluable<InType>::ConstPtr T_iv_;
};

template <int DIM>
auto PreIntVelocityStateVar<DIM>::MakeShared(const T& value, const Evaluable<InType>::ConstPtr& T_iv, const std::string& name)
    -> Ptr {
  return std::make_shared<PreIntVelocityStateVar<DIM>>(value, T_iv, name);
}

template <int DIM>
PreIntVelocityStateVar<DIM>::PreIntVelocityStateVar(const T& value, const Evaluable<InType>::ConstPtr& T_iv, const std::string& name)
    : Base(value, DIM, name), T_iv_(T_iv) {}

template <int DIM>
bool PreIntVelocityStateVar<DIM>::update(const Eigen::VectorXd& perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error(
        "PreIntVelocityStateVar::update: perturbation size mismatch");
  //
  const Eigen::Matrix3d C_iv = T_iv_->value().matrix().block<3, 3>(0, 0);
  this->value_ = this->value_ + C_iv * perturbation;
  return true;
}

template <int DIM>
StateVarBase::Ptr PreIntVelocityStateVar<DIM>::clone() const {
  return std::make_shared<PreIntVelocityStateVar<DIM>>(*this);
}

}  // namespace vspace
}  // namespace steam