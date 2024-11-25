#include "steam/evaluable/se3/se3_state_var_global_perturb.hpp"

namespace steam {
namespace se3 {

auto SE3StateVarGlobalPerturb::MakeShared(const T &value, const std::string &name) -> Ptr {
  return std::make_shared<SE3StateVarGlobalPerturb>(value, name);
}

SE3StateVarGlobalPerturb::SE3StateVarGlobalPerturb(const T &value, const std::string &name)
    : Base(value, 6, name) {}

bool SE3StateVarGlobalPerturb::update(const Eigen::VectorXd &perturbation) {
  if (perturbation.size() != this->perturb_dim())
    throw std::runtime_error("SE3StateVarGlobalPerturb::update: perturbation size mismatch");
  const auto delta_r = perturbation.block<3, 1>(0, 0);
  const auto delta_phi = perturbation.block<3, 1>(3, 0);
  Eigen::Matrix4d T_iv = value_.matrix();
  Eigen::Matrix3d C_iv = T_iv.block<3, 3>(0, 0);
  Eigen::Vector3d r_vi_in_i = T_iv.block<3, 1>(0, 3);
  r_vi_in_i += C_iv * delta_r;
  C_iv = C_iv * lgmath::so3::vec2rot(delta_phi);
  T_iv.block<3, 3>(0, 0) = C_iv;
  T_iv.block<3, 1>(0, 3) = r_vi_in_i;
  value_ = T(T_iv);
  return true;
}

StateVarBase::Ptr SE3StateVarGlobalPerturb::clone() const {
  return std::make_shared<SE3StateVarGlobalPerturb>(*this);
}

}  // namespace se3
}  // namespace steam