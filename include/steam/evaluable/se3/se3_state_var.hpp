#pragma once

#include "lgmath.hpp"

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace se3 {

class SE3StateVar : public StateVar<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<SE3StateVar>;
  using ConstPtr = std::shared_ptr<const SE3StateVar>;

  using T = lgmath::se3::Transformation;
  using Base = StateVar<T>;

  static Ptr MakeShared(const T& value, const std::string& name = "");
  SE3StateVar(const T& value, const std::string& name = "");

  bool update(const Eigen::VectorXd& perturbation) override;
  StateVarBase::Ptr clone() const override;
};

}  // namespace se3
}  // namespace steam