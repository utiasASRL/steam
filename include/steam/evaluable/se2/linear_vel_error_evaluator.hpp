#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se2 {

class LinearVelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 2, 1>> {
 public:
  using Ptr = std::shared_ptr<LinearVelErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const LinearVelErrorEvaluator>;

  using InType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 2, 1>;

  static Ptr MakeShared(const Eigen::Vector2d& vel_meas,
                        const Evaluable<InType>::ConstPtr &w_iv_inv);
  LinearVelErrorEvaluator(const Eigen::Vector2d& vel_meas,
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
  Eigen::Matrix<double, 2, 3> D_ = Eigen::Matrix<double, 2, 3>::Zero();
};

LinearVelErrorEvaluator::Ptr linearVelError(
    const Eigen::Vector2d& vel_meas,
    const Evaluable<LinearVelErrorEvaluator::InType>::ConstPtr &w_iv_inv);

}  // namespace se2
}  // namespace steam