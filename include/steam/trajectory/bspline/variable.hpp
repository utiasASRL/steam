#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace bspline {

class Variable {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<Variable>;
  using ConstPtr = std::shared_ptr<const Variable>;

  static Ptr MakeShared(const Time& time,
                        const vspace::VSpaceStateVar<6>::Ptr& c) {
    return std::make_shared<Variable>(time, c);
  }

  Variable(const Time& time, const vspace::VSpaceStateVar<6>::Ptr& c)
      : time_(time), c_(c) {}

  virtual ~Variable() = default;

  /** \brief Get timestamp */
  const Time& getTime() const { return time_; }

  /** \brief Get pose evaluator */
  const vspace::VSpaceStateVar<6>::Ptr& getC() const { return c_; }

 private:
  /** \brief Timestamp of trajectory variable */
  Time time_;

  /** \brief c vector evaluator */
  const vspace::VSpaceStateVar<6>::Ptr c_;
};

}  // namespace bspline
}  // namespace traj
}  // namespace steam