#pragma once

#include "lgmath.hpp"

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace se3 {

class SE3StateVarGlobalPerturb : public StateVar<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<SE3StateVarGlobalPerturb>;
  using ConstPtr = std::shared_ptr<const SE3StateVarGlobalPerturb>;

  using T = lgmath::se3::Transformation;
  using Base = StateVar<T>;

  static Ptr MakeShared(const T& value, const std::string& name = "");
  SE3StateVarGlobalPerturb(const T& value, const std::string& name = "");

  /** \brief updates state using the given perturbation
   * C <- C * exp(delta_phi^), r <- r + C * delta_r
   * \param perturbation 6x1 vector containing delta_r, delta_phi
  */
  bool update(const Eigen::VectorXd& perturbation) override;
  StateVarBase::Ptr clone() const override;
};

}  // namespace se3
}  // namespace steam