#include "steam/evaluable/stereo/homo_point_state_var.hpp"

namespace steam {
namespace stereo {

auto HomoPointStateVar::MakeShared(const Eigen::Vector3d& value, bool scale,
                                   const std::string& name) -> Ptr {
  return std::make_shared<HomoPointStateVar>(value, scale, name);
}

HomoPointStateVar::HomoPointStateVar(const Eigen::Vector3d& value, bool scale,
                                     const std::string& name)
    : Base(Eigen::Vector4d::Constant(1.0), 3, name), scale_(scale) {
  this->value_.head<3>() = value;
  this->refreshHomoScaling();
}

bool HomoPointStateVar::update(const Eigen::VectorXd& perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error(
        "HomoPointStateVar::update: perturbation size mismatch");

  this->value_.head<3>() += perturbation;
  this->refreshHomoScaling();

  return true;
}

StateVarBase::Ptr HomoPointStateVar::clone() const {
  return std::make_shared<HomoPointStateVar>(*this);
}

void HomoPointStateVar::refreshHomoScaling() {
  if (!scale_) return;
  const double mag = this->value_.head<3>().norm();
  if (mag == 0.0)
    this->value_(3) = 1.0;
  else
    this->value_ /= mag;
}

}  // namespace stereo
}  // namespace steam
