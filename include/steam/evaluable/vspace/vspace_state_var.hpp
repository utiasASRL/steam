#pragma once

#include "lgmath.hpp"

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace vspace {

template <int DIM = Eigen::Dynamic>
class VSpaceStateVar : public StateVar<Eigen::Matrix<double, DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<VSpaceStateVar>;
  using ConstPtr = std::shared_ptr<const VSpaceStateVar>;

  using T = Eigen::Matrix<double, DIM, 1>;
  using Base = StateVar<T>;

  static Ptr MakeShared(const T& value);
  VSpaceStateVar(const T& value);

  bool update(const Eigen::VectorXd& perturbation) override;
  StateVarBase::Ptr clone() const override;

  T value() const override;
  typename Node<T>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const typename Node<T>::Ptr& node,
                Jacobians& jacs) const override;
};

template <int DIM>
auto VSpaceStateVar<DIM>::MakeShared(const T& value) -> Ptr {
  return std::make_shared<VSpaceStateVar<DIM>>(value);
}

template <int DIM>
VSpaceStateVar<DIM>::VSpaceStateVar(const T& value) : Base(value, DIM) {}

template <int DIM>
bool VSpaceStateVar<DIM>::update(const Eigen::VectorXd& perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error(
        "VSpaceStateVar::update: perturbation size mismatch");
  //
  this->value_ = this->value_ + perturbation;
  return true;
}

template <int DIM>
StateVarBase::Ptr VSpaceStateVar<DIM>::clone() const {
  return std::make_shared<VSpaceStateVar<DIM>>(*this);
}

template <int DIM>
auto VSpaceStateVar<DIM>::value() const -> T {
  return this->value_;
}

template <int DIM>
auto VSpaceStateVar<DIM>::forward() const -> typename Node<T>::Ptr {
  return Node<T>::MakeShared(this->value_);
}

template <int DIM>
void VSpaceStateVar<DIM>::backward(const Eigen::MatrixXd& lhs,
                                   const typename Node<T>::Ptr& node,
                                   Jacobians& jacs) const {
  if (this->active()) jacs.add(this->key(), lhs);
}

}  // namespace vspace
}  // namespace steam