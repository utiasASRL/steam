/**
 * \file geman_mcclure_loss_func.hpp
 * \author Sean Anderson, ASRL
 */
#pragma once

#include "steam/problem/loss_func/base_loss_func.hpp"

namespace steam {

/** \brief Geman-McClure loss function class */
class GemanMcClureLossFunc : public BaseLossFunc {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<GemanMcClureLossFunc>;
  using ConstPtr = std::shared_ptr<const GemanMcClureLossFunc>;

  static Ptr MakeShared(double k) {
    return std::make_shared<GemanMcClureLossFunc>(k);
  }

  /**
   * \brief Constructor -- k is the `threshold' based on number of std devs
   * (1-3 is typical)
   */
  GemanMcClureLossFunc(double k) : k2_(k * k) {}

  /** \brief Cost function (basic evaluation of the loss function) */
  double cost(double whitened_error_norm) const override {
    double e2 = whitened_error_norm * whitened_error_norm;
    return 0.5 * e2 / (k2_ + e2);
  }

  /**
   * \brief Weight for iteratively reweighted least-squares (influence function
   * div. by error)
   */
  double weight(double whitened_error_norm) const override {
    double e2 = whitened_error_norm * whitened_error_norm;
    double k2e2 = k2_ + e2;
    return k2_ * k2_ / (k2e2 * k2e2);
  }

 private:
  /** \brief GM constant */
  double k2_;
};

}  // namespace steam
