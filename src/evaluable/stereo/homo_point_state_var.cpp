#include "steam/evaluable/stereo/homo_point_state_var.hpp"

namespace steam {
namespace stereo {

auto HomoPointStateVar::MakeShared(const Eigen::Vector3d& value) -> Ptr {
  return std::make_shared<HomoPointStateVar>(value);
}

HomoPointStateVar::HomoPointStateVar(const Eigen::Vector3d& value)
    : Base(Eigen::Vector4d::Constant(1.0), 3) {
  this->value_.head<3>() = value;
  this->refreshHomogeneousScaling();
}

bool HomoPointStateVar::update(const Eigen::VectorXd& perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error(
        "HomoPointStateVar::update: perturbation size mismatch");

  this->value_.head<3>() += perturbation;
  this->refreshHomogeneousScaling();

  return true;
}

StateVarBase::Ptr HomoPointStateVar::clone() const {
  return std::make_shared<HomoPointStateVar>(*this);
}

auto HomoPointStateVar::forward() const -> Node<T>::Ptr {
  return Node<T>::MakeShared(this->value_);
}

void HomoPointStateVar::backward(const Eigen::MatrixXd& lhs,
                                 const Node<T>::Ptr& node,
                                 Jacobians& jacs) const {
  if (this->active()) jacs.add(this->key(), lhs);
}

void HomoPointStateVar::refreshHomogeneousScaling() {
  // Get length of xyz-portion of landmark
  const double invmag = 1.0 / this->value_.head<3>().norm();

  // Update xyz-portion to be unit-length
  this->value_.head<3>() *= invmag;

  // Update scaling
  this->value_[3] *= invmag;
}

}  // namespace stereo
}  // namespace steam
