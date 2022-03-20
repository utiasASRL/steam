#pragma once

#include "steam/blockmat/BlockSparseMatrix.hpp"
#include "steam/blockmat/BlockVector.hpp"
#include "steam/problem2/state_vec.hpp"

namespace steam {

/**
 * \brief Interface for a 'cost term' class that contributes to the objective
 * function.
 * \note Functions must be provided to calculate the scalar 'cost' and to build
 * the contributions to the Gauss-Newton system of equations.
 */
class CostTermBase {
 public:
  using Ptr = std::shared_ptr<CostTermBase>;
  using ConstPtr = std::shared_ptr<const CostTermBase>;

  virtual ~CostTermBase() = default;

  /** \brief Compute the cost to the objective function */
  virtual double cost() const = 0;

  /**
   * \brief Add the contribution of this cost term to the left-hand (Hessian)
   * and right-hand (gradient vector) sides of the Gauss-Newton system of
   * equations.
   */
  virtual void buildGaussNewtonTerms(const StateVec &state_vec,
                                     BlockSparseMatrix *approximate_hessian,
                                     BlockVector *gradient_vector) const = 0;
};

}  // namespace steam
