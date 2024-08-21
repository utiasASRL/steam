#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace imu {

class DMIErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<DMIErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const DMIErrorEvaluator>;

  using VelInType = Eigen::Matrix<double, 6, 1>;
  using DMIInType = double;
  using ScaleInType = Eigen::Matrix<double, 1, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;
  using Time = steam::traj::Time;
  using JacType = Eigen::Matrix<double, 1, 6>;

  static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<ScaleInType>::ConstPtr &scale,
                        const DMIInType &dmi_meas);
  DMIErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
                    const Evaluable<ScaleInType>::ConstPtr &scale,
                    const DMIInType &dmi_meas);

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
      throw std::runtime_error("DMI measurement time was not initialized");
  }

 private:
  // evaluable
  const Evaluable<VelInType>::ConstPtr velocity_;
  const Evaluable<ScaleInType>::ConstPtr scale_;
  const DMIInType dmi_meas_;
  JacType jac_vel_ = JacType::Zero();
  Eigen::Matrix<double, 1, 1> jac_scale_ = Eigen::Matrix<double, 1, 1>::Zero();
  Time time_;
  bool time_init_ = false;
};

DMIErrorEvaluator::Ptr DMIError(
    const Evaluable<DMIErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<DMIErrorEvaluator::ScaleInType>::ConstPtr &scale,
    const DMIErrorEvaluator::DMIInType &dmi_meas);

}  // namespace imu
}  // namespace steam