#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class VelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<VelErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const VelErrorEvaluator>;

  using InType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Eigen::Vector2d vel_meas,
                        const Evaluable<InType>::ConstPtr &w_iv_inv);
  VelErrorEvaluator(const Eigen::Vector2d vel_meas,
                        const Evaluable<InType>::ConstPtr &w_iv_inv);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

 private:
  // evaluable
  const Evaluable<InType>::ConstPtr w_iv_inv_;
  // constants
  const Eigen::Vector2d vel_meas_;
  Eigen::Matrix<double, 3, 6> D_;
};

VelErrorEvaluator::Ptr velError(
    const Eigen::Vector2d vel_meas,
    const Evaluable<VelErrorEvaluator::InType>::ConstPtr &w_iv_inv);

}  // namespace p2p
}  // namespace steam