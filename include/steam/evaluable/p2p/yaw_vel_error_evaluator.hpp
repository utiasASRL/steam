#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class YawVelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<YawVelErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const YawVelErrorEvaluator>;

  using InType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;

  static Ptr MakeShared(Eigen::Matrix<double, 1, 1> vel_meas,
                        const Evaluable<InType>::ConstPtr &w_iv_inv);
  YawVelErrorEvaluator(Eigen::Matrix<double, 1, 1> vel_meas,
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
  const Eigen::Matrix<double, 1, 1> vel_meas_;
  Eigen::Matrix<double, 1, 6> D_; // pick out yaw vel
};

YawVelErrorEvaluator::Ptr velError(
    const Eigen::Matrix<double, 1, 1> vel_meas,
    const Evaluable<YawVelErrorEvaluator::InType>::ConstPtr &w_iv_inv);

}  // namespace p2p
}  // namespace steam