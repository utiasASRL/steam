/**
 * \file cauchy_loss_func.hpp
 * \author Sean Anderson, ASRL
 */
#pragma once

#include "steam/problem/loss_func/base_loss_func.hpp"

namespace steam {

/** \brief Cauchy loss function class */
class CauchyLossFunc : public BaseLossFunc {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<CauchyLossFunc>;
  using ConstPtr = std::shared_ptr<const CauchyLossFunc>;

  static Ptr MakeShared(double k) {
    return std::make_shared<CauchyLossFunc>(k);
  }

  /**
   * \brief Constructor -- k is the `threshold' based on number of std devs
   * (1-3 is typical)
   */
  CauchyLossFunc(double k) : k_(k) {}

  /** \brief Cost function (basic evaluation of the loss function) */
  double cost(double whitened_error_norm) const override {
    double e_div_k = fabs(whitened_error_norm) / k_;
    return 0.5 * k_ * k_ * std::log(1.0 + e_div_k * e_div_k);
  }

  /**
   * \brief Weight for iteratively reweighted least-squares (influence function
   * div. by error)
   */
  double weight(double whitened_error_norm) const override {
    double e_div_k = fabs(whitened_error_norm) / k_;
    return 1.0 / (1.0 + e_div_k * e_div_k);
  }

 private:
  /** \brief Cauchy constant */
  double k_;
};

}  // namespace steam
