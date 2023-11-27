#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace imu {

class GyroErrorEvaluatorSE2 : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<GyroErrorEvaluatorSE2>;
  using ConstPtr = std::shared_ptr<const GyroErrorEvaluatorSE2>;

  using VelInType = Eigen::Matrix<double, 6, 1>;
  using BiasInType = Eigen::Matrix<double, 1, 1>;
  using ImuInType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;
  using Time = steam::traj::Time;
  using JacType = Eigen::Matrix<double, 1, 6>;

  static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const ImuInType &gyro_meas);
  // gyro_meas: ax,ay,az,wx,wy,wz
  GyroErrorEvaluatorSE2(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const ImuInType &gyro_meas);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

  void setTime(Time time) {
    time_ = time;
    time_init_ = true;
  };

  Time getTime() const {
    if (time_init_)
      return time_;
    else
      throw std::runtime_error("Gyro measurement time was not initialized");
  }

 private:
  // evaluable
  const Evaluable<VelInType>::ConstPtr velocity_;
  const Evaluable<BiasInType>::ConstPtr bias_;
  const ImuInType gyro_meas_;
  JacType jac_vel_ = JacType::Zero();
  Eigen::Matrix<double, 1, 1> jac_bias_ =
      Eigen::Matrix<double, 1, 1>::Ones() * -1;
  Time time_;
  bool time_init_ = false;
};

GyroErrorEvaluatorSE2::Ptr GyroErrorSE2(
    const Evaluable<GyroErrorEvaluatorSE2::VelInType>::ConstPtr &velocity,
    const Evaluable<GyroErrorEvaluatorSE2::BiasInType>::ConstPtr &bias,
    const GyroErrorEvaluatorSE2::ImuInType &gyro_meas);

}  // namespace imu
}  // namespace steam