#pragma once

#include "lgmath.hpp"

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace se2 {

class SE2StateVar : public StateVar<lgmath::se2::Transformation> {
 public:
  using Ptr = std::shared_ptr<SE2StateVar>;
  using ConstPtr = std::shared_ptr<const SE2StateVar>;

  using T = lgmath::se2::Transformation;
  using Base = StateVar<T>;

  static Ptr MakeShared(const T& value, const std::string& name = "");
  SE2StateVar(const T& value, const std::string& name = "");

  bool update(const Eigen::VectorXd& perturbation) override;
  StateVarBase::Ptr clone() const override;
};

}  // namespace se2
}  // namespace steam