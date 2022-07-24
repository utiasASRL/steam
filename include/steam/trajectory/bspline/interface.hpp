#pragma once

#include <Eigen/Core>

#include "steam/problem/cost_term/weighted_least_sq_cost_term.hpp"
#include "steam/problem/optimization_problem.hpp"
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

  void addStateVariables(Problem& problem) const;

  using KnotMap = std::map<Time, Variable::Ptr>;
  const KnotMap& knot_map() const { return knot_map_; }

 protected:
  const Time knot_spacing_;
  /** \brief Ordered map of knots */
  KnotMap knot_map_;
};

}  // namespace bspline
}  // namespace traj
}  // namespace steam