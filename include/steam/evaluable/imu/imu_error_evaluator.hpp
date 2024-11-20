#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace imu {

class IMUErrorEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<IMUErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const IMUErrorEvaluator>;

  using PoseInType = lgmath::se3::Transformation;
  using VelInType = Eigen::Matrix<double, 6, 1>;
  using AccInType = Eigen::Matrix<double, 6, 1>;
  using BiasInType = Eigen::Matrix<double, 6, 1>;
  using ImuInType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &transform,
                        const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<AccInType>::ConstPtr &acceleration,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                        const ImuInType &imu_meas);
  // imu_meas: ax,ay,az,wx,wy,wz
  IMUErrorEvaluator(const Evaluable<PoseInType>::ConstPtr &transform,
                    const Evaluable<VelInType>::ConstPtr &velocity,
                    const Evaluable<AccInType>::ConstPtr &acceleration,
                    const Evaluable<BiasInType>::ConstPtr &bias,
                    const Evaluable<PoseInType>::ConstPtr &transform_i_to_m,
                    const ImuInType &imu_meas);

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
  const Evaluable<VelInType>::ConstPtr velocity_;
  const Evaluable<AccInType>::ConstPtr acceleration_;
  const Evaluable<BiasInType>::ConstPtr bias_;
  const Evaluable<PoseInType>::ConstPtr transform_i_to_m_;
  const ImuInType imu_meas_;
  Eigen::Matrix<double, 3, 1> gravity_ = Eigen::Matrix<double, 3, 1>::Zero();
};

IMUErrorEvaluator::Ptr imuError(
    const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform,
    const Evaluable<IMUErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<IMUErrorEvaluator::AccInType>::ConstPtr &acceleration,
    const Evaluable<IMUErrorEvaluator::BiasInType>::ConstPtr &bias,
    const Evaluable<IMUErrorEvaluator::PoseInType>::ConstPtr &transform_i_to_m,
    const IMUErrorEvaluator::ImuInType &imu_meas);

}  // namespace imu
}  // namespace steam