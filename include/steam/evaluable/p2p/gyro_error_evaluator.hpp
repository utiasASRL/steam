#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class GyroErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<GyroErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const GyroErrorEvaluator>;

  using PoseInType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Eigen::Vector3d &gyro_meas,
                        const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
                        const Evaluable<PoseInType>::ConstPtr &T_ms_curr);

  GyroErrorEvaluator(const Eigen::Vector3d &gyro_meas, 
                     const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
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
  const Eigen::Vector3d gyro_meas_;
};;

GyroErrorEvaluator::Ptr gyroError(const Eigen::Vector3d &gyro_meas,
                                  const Evaluable<GyroErrorEvaluator::PoseInType>::ConstPtr &T_ms_prev,
                                  const Evaluable<GyroErrorEvaluator::PoseInType>::ConstPtr &T_ms_curr);

}  // namespace p2p
}  // namespace steam