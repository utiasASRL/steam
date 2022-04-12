/**
 * \file dcs_loss_func.hpp
 * \author Sean Anderson, ASRL
 */
#pragma once

#include "steam/problem/loss_func/base_loss_func.hpp"

namespace steam {

/** \brief Huber loss function class */
class DcsLossFunc : public BaseLossFunc {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<DcsLossFunc>;
  using ConstPtr = std::shared_ptr<const DcsLossFunc>;

  static Ptr MakeShared(double k) { return std::make_shared<DcsLossFunc>(k); }

  /**
   * \brief Constructor -- k is the `threshold' based on number of std devs
   * (1-3 is typical)
   */
  DcsLossFunc(double k) : k2_(k * k) {}

  /** \brief Cost function (basic evaluation of the loss function) */
  double cost(double whitened_error_norm) const override {
    double e2 = whitened_error_norm * whitened_error_norm;
    if (e2 <= k2_) {
      return 0.5 * e2;
    } else {
      return 2.0 * k2_ * e2 / (k2_ + e2) - 0.5 * k2_;
    }
  }

  /**
   * \brief Weight for iteratively reweighted least-squares (influence function
   * div. by error)
   */
  double weight(double whitened_error_norm) const override {
    double e2 = whitened_error_norm * whitened_error_norm;
    if (e2 <= k2_) {
      return 1.0;
    } else {
      double k2e2 = k2_ + e2;
      return 4.0 * k2_ * k2_ / (k2e2 * k2e2);
    }
  }

 private:
  /** \brief DCS constant */
  double k2_;
};

}  // namespace steam
