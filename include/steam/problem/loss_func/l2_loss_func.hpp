/**
 * \file l2_loss_func.hpp
 * \author Sean Anderson, ASRL
 */
#pragma once

#include "steam/problem/loss_func/base_loss_func.hpp"

namespace steam {

/** \brief 'L2' loss function */
class L2LossFunc : public BaseLossFunc {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<L2LossFunc>;
  using ConstPtr = std::shared_ptr<const L2LossFunc>;

  static Ptr MakeShared() { return std::make_shared<L2LossFunc>(); }

  /** \brief Constructor */
  L2LossFunc() = default;

  /** \brief Cost function (basic evaluation of the loss function) */
  double cost(double whitened_error_norm) const override {
    return 0.5 * whitened_error_norm * whitened_error_norm;
  }

  /**
   * \brief Weight for iteratively reweighted least-squares (influence function
   * div. by error)
   */
  double weight(double whitened_error_norm) const override { return 1.0; }
};

}  // namespace steam
