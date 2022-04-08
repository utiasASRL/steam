#include "steam/evaluable/se3/se3_state_var.hpp"

namespace steam {
namespace se3 {

auto SE3StateVar::MakeShared(const T &value) -> Ptr {
  return std::make_shared<SE3StateVar>(value);
}

SE3StateVar::SE3StateVar(const T &value) : Base(value, 6) {}

bool SE3StateVar::update(const Eigen::VectorXd &perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error("SE3StateVar::update: perturbation size mismatch");
  // Update the Lie matrix using a left-multiplicative perturbation
  value_ = T(perturbation) * value_;
  return true;
}

StateVarBase::Ptr SE3StateVar::clone() const {
  return std::make_shared<SE3StateVar>(*this);
}

auto SE3StateVar::value() const -> T { return value_; }

auto SE3StateVar::forward() const -> Node<T>::Ptr {
  return Node<T>::MakeShared(this->value_);
}

void SE3StateVar::backward(const Eigen::MatrixXd &lhs, const Node<T>::Ptr &node,
                           Jacobians &jacs) const {
  if (this->active()) jacs.add(this->key(), lhs);
}

}  // namespace se3
}  // namespace steam