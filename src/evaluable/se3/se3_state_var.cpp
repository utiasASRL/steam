#include "steam/evaluable/se3/se3_state_var.hpp"

namespace steam {
namespace se3 {

auto SE3StateVar::MakeShared(const T &value, const std::string &name) -> Ptr {
  return std::make_shared<SE3StateVar>(value, name);
}

SE3StateVar::SE3StateVar(const T &value, const std::string &name)
    : Base(value, 6, name) {}

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

}  // namespace se3
}  // namespace steam