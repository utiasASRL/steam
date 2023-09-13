#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace imu {

class GyroErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<GyroErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const GyroErrorEvaluator>;

  using VelInType = Eigen::Matrix<double, 6, 1>;
  using BiasInType = Eigen::Matrix<double, 6, 1>;
  using ImuInType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const ImuInType &gyro_meas);
  // gyro_meas: ax,ay,az,wx,wy,wz
  GyroErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
                     const Evaluable<BiasInType>::ConstPtr &bias,
                     const ImuInType &gyro_meas);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

 private:
  // evaluable
  const Evaluable<VelInType>::ConstPtr velocity_;
  const Evaluable<BiasInType>::ConstPtr bias_;
  const ImuInType gyro_meas_;
  Eigen::Matrix<double, 3, 6> Dw_ = Eigen::Matrix<double, 3, 6>::Zero();
};

GyroErrorEvaluator::Ptr GyroError(
    const Evaluable<GyroErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<GyroErrorEvaluator::BiasInType>::ConstPtr &bias,
    const GyroErrorEvaluator::ImuInType &gyro_meas);

}  // namespace imu
}  // namespace steam