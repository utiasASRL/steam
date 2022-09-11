/**
 * \file base_loss_func.hpp
 * \author Sean Anderson, ASRL
 */
#pragma once

#include <memory>

namespace steam {

/**
 * \brief Base loss function class.
 * A loss function must implement both the cost and weight functions. For
 * example, the basic least-square L2 loss function has: cost: e^2, and
 * weight: 1.
 */
class BaseLossFunc {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<BaseLossFunc>;
  using ConstPtr = std::shared_ptr<const BaseLossFunc>;

  /** \brief Destructor */
  virtual ~BaseLossFunc() = default;

  /** \brief Cost function (basic evaluation of the loss function) */
  virtual double cost(double whitened_error_norm) const = 0;

  /**
   * \brief Weight for iteratively reweighted least-squares (influence function
   * div. by error)
   */
  virtual double weight(double whitened_error_norm) const = 0;
};

}  // namespace steam
