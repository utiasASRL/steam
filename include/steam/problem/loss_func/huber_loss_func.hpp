/**
 * \file huber_loss_func.hpp
 * \author Sean Anderson, ASRL
 */
#pragma once

#include "steam/problem/loss_func/base_loss_func.hpp"

namespace steam {

/** \brief Huber loss function class */
class HuberLossFunc : public BaseLossFunc {
 public:
  /// Convenience typedefs
  using Ptr = std::shared_ptr<HuberLossFunc>;
  using ConstPtr = std::shared_ptr<const HuberLossFunc>;

  static Ptr MakeShared(double k) { return std::make_shared<HuberLossFunc>(k); }

  /**
   * \brief Constructor -- k is the `threshold' based on number of std devs
   * (1-3 is typical)
   */
  HuberLossFunc(double k) : k_(k) {}

  /** \brief Cost function (basic evaluation of the loss function) */
  double cost(double whitened_error_norm) const override {
    double e2 = whitened_error_norm * whitened_error_norm;
    double abse = fabs(whitened_error_norm);
    if (abse <= k_) {
      return 0.5 * e2;
    } else {
      return k_ * (abse - 0.5 * k_);
    }
  }

  /**
   * \brief Weight for iteratively reweighted least-squares (influence function
   * div. by error)
   */
  double weight(double whitened_error_norm) const override {
    double abse = fabs(whitened_error_norm);
    if (abse <= k_) {
      return 1.0;
    } else {
      return k_ / abse;
    }
  }

 private:
  /** \brief Huber constant */
  double k_;
};

}  // namespace steam
