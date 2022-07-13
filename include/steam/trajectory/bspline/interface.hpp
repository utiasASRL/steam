#pragma once

#include <Eigen/Core>

#include "steam/problem/OptimizationProblem.hpp"
#include "steam/problem/cost_term/weighted_least_sq_cost_term.hpp"
#include "steam/trajectory/bspline/variable.hpp"
#include "steam/trajectory/interface.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace bspline {

class Interface : public traj::Interface {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<Interface>;
  using ConstPtr = std::shared_ptr<const Interface>;

  using VeloType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Time& knot_spacing = Time(0.1));
  Interface(const Time& knot_spacing = Time(0.1));

  /** \brief Get velocity evaluator */
  Evaluable<VeloType>::ConstPtr getVelocityInterpolator(const Time& time);

  void addPriorCostTerms(OptimizationProblem& problem) const override {}
  void addStateVariables(OptimizationProblem& problem) const;

  void setActiveWindow(
      const Time& start,
      const Time& end = Time(std::numeric_limits<int64_t>::max()));

 protected:
  /** \brief Ordered map of knots */
  const Time knot_spacing_;
  /** \brief Ordered map of knots */
  std::map<Time, Variable::Ptr> knot_map_;
  std::map<Time, Variable::Ptr> active_knot_map_;
};

}  // namespace bspline
}  // namespace traj
}  // namespace steam