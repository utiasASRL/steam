#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class YawErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<YawErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const YawErrorEvaluator>;

  using PoseInType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 1, 1>;

  static Ptr MakeShared(const double yaw_meas,
                        const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
                        const Evaluable<PoseInType>::ConstPtr &T_ms_curr
                        );
  YawErrorEvaluator(const double yaw_meas, const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
                    const Evaluable<PoseInType>::ConstPtr &T_ms_curr);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

 private:
  // evaluable
  const Evaluable<PoseInType>::ConstPtr T_ms_prev_;
  const Evaluable<PoseInType>::ConstPtr T_ms_curr_;
  // constants
  const double yaw_meas_;
  Eigen::Matrix<double, 1, 3> d_;
};

YawErrorEvaluator::Ptr yawError(
    const double yaw_meas, const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_prev,
    const Evaluable<YawErrorEvaluator::PoseInType>::ConstPtr &T_ms_curr);

}  // namespace p2p
}  // namespace steam