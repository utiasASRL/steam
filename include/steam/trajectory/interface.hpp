#pragma once

#include <memory>

#include "steam/problem/OptimizationProblem.hpp"

namespace steam {
namespace traj {

class Interface {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<Interface>;
  using ConstPtr = std::shared_ptr<const Interface>;

  virtual ~Interface() = default;

  /**
   * \brief Get binary cost terms associated with the prior for active parts of
   * the trajectory
   */
  virtual void addPriorCostTerms(OptimizationProblem& problem) const = 0;
};

}  // namespace traj
}  // namespace steam