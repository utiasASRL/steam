#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace imu {

class DopplerErrorEvaluatorSE2 : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<DopplerErrorEvaluatorSE2>;
  using ConstPtr = std::shared_ptr<const DopplerErrorEvaluatorSE2>;

  using VelInType = Eigen::Matrix<double, 6, 1>;
  using BiasInType = Eigen::Matrix<double, 6, 1>;
  using AzimuthInType = double;
  using DopplerInType = Eigen::Matrix<double, 1, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;
  using Time = steam::traj::Time;
  using JacType = Eigen::Matrix<double, 1, 6>;

  static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const AzimuthInType &azimuth,
                        const DopplerInType &doppler_meas);
  DopplerErrorEvaluatorSE2(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const AzimuthInType &azimuth,
                        const DopplerInType &doppler_meas);

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
  const DopplerInType doppler_meas_;
  Eigen::Matrix<double, 1, 6> D_ = Eigen::Matrix<double, 1, 6>::Zero();
};

DopplerErrorEvaluatorSE2::Ptr DopplerErrorSE2(
    const Evaluable<DopplerErrorEvaluatorSE2::VelInType>::ConstPtr &velocity,
    const Evaluable<DopplerErrorEvaluatorSE2::BiasInType>::ConstPtr &bias,
    const DopplerErrorEvaluatorSE2::AzimuthInType &azimuth,
    const DopplerErrorEvaluatorSE2::DopplerInType &doppler_meas);

}  // namespace imu
}  // namespace steam