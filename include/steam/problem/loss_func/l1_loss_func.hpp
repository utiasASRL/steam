/**
 * \file l1_loss_func.hpp
 * \author ASRL
 */
#pragma once

#include "steam/problem/loss_func/base_loss_func.hpp"

namespace steam {

/** \brief 'L1' loss function */
class L1LossFunc : public BaseLossFunc {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<L1LossFunc>;
  using ConstPtr = std::shared_ptr<const L1LossFunc>;

  static Ptr MakeShared() { return std::make_shared<L1LossFunc>(); }

  /** \brief Constructor */
  L1LossFunc() = default;

  /** \brief Cost function (basic evaluation of the loss function) */
  double cost(double whitened_error_norm) const override {
    return fabs(whitened_error_norm);
  }

  /**
   * \brief Weight for iteratively reweighted least-squares (influence function
   * div. by error)
   */
  double weight(double whitened_error_norm) const override {
    return 1.0 / fabs(whitened_error_norm);
  }
};

}  // namespace steam
