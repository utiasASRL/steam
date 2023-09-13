#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace imu {

class AccelerationErrorEvaluator
    : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<AccelerationErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const AccelerationErrorEvaluator>;

  using PoseInType = lgmath::se3::Transformation;
  using AccInType = Eigen::Matrix<double, 6, 1>;
  using BiasInType = Eigen::Matrix<double, 6, 1>;
  using ImuInType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                        const Evaluable<AccInType>::ConstPtr &acceleration,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                        const ImuInType &acc_meas);
  // acc_meas: ax,ay,az,wx,wy,wz
  AccelerationErrorEvaluator(
      const Evaluable<PoseInType>::ConstPtr &transform,
      const Evaluable<AccInType>::ConstPtr &acceleration,
      const Evaluable<BiasInType>::ConstPtr &bias,
      const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
      const ImuInType &acc_meas);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

  void setGravity(double gravity) { gravity_(2, 0) = gravity; }

 private:
  // evaluable
  const Evaluable<PoseInType>::ConstPtr transform_;
  const Evaluable<AccInType>::ConstPtr acceleration_;
  const Evaluable<BiasInType>::ConstPtr bias_;
  const Evaluable<PoseInType>::ConstPtr transform_i_to_m_;
  const ImuInType acc_meas_;
  Eigen::Matrix<double, 3, 6> Da_ = Eigen::Matrix<double, 3, 6>::Zero();
  Eigen::Matrix<double, 3, 6> Dw_ = Eigen::Matrix<double, 3, 6>::Zero();
  Eigen::Matrix<double, 3, 1> gravity_ = Eigen::Matrix<double, 3, 1>::Zero();
};

AccelerationErrorEvaluator::Ptr AccelerationError(
    const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr
        &transform,
    const Evaluable<AccelerationErrorEvaluator::AccInType>::ConstPtr
        &acceleration,
    const Evaluable<AccelerationErrorEvaluator::BiasInType>::ConstPtr &bias,
    const Evaluable<AccelerationErrorEvaluator::PoseInType>::ConstPtr
        &transform_i_to_m,
    const AccelerationErrorEvaluator::ImuInType &acc_meas);

}  // namespace imu
}  // namespace steam