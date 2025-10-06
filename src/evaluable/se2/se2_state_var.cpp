#include "steam/evaluable/se2/se2_state_var.hpp"

namespace steam {
namespace se2 {

auto SE2StateVar::MakeShared(const T &value, const std::string &name) -> Ptr {
  return std::make_shared<SE2StateVar>(value, name);
}

SE2StateVar::SE2StateVar(const T &value, const std::string &name)
    : Base(value, 3, name) {}

bool SE2StateVar::update(const Eigen::VectorXd &perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error("SE2StateVar::update: perturbation size mismatch");
  // Update the Lie matrix using a left-multiplicative perturbation
  value_ = T(perturbation) * value_;
  return true;
}

StateVarBase::Ptr SE2StateVar::clone() const {
  return std::make_shared<SE2StateVar>(*this);
}

}  // namespace se2
}  // namespace steam