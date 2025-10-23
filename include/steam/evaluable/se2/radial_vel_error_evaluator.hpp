#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace se2 {

class RadialVelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<RadialVelErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const RadialVelErrorEvaluator>;

  using VelInType = Eigen::Matrix<double, 3, 1>;
  using BiasInType = Eigen::Matrix<double, 3, 1>;
  using AzimuthInType = double;
  using DopplerInType = Eigen::Matrix<double, 1, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;
  using Time = steam::traj::Time;
  using JacType = Eigen::Matrix<double, 1, 3>;

  static Ptr MakeShared(const Evaluable<VelInType>::ConstPtr &velocity,
                        const Evaluable<BiasInType>::ConstPtr &bias,
                        const AzimuthInType &azimuth,
                        const DopplerInType &doppler_meas);
  RadialVelErrorEvaluator(const Evaluable<VelInType>::ConstPtr &velocity,
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
  Eigen::Matrix<double, 1, 3> D_ = Eigen::Matrix<double, 1, 3>::Zero();
};

RadialVelErrorEvaluator::Ptr radialVelError(
    const Evaluable<RadialVelErrorEvaluator::VelInType>::ConstPtr &velocity,
    const Evaluable<RadialVelErrorEvaluator::BiasInType>::ConstPtr &bias,
    const RadialVelErrorEvaluator::AzimuthInType &azimuth,
    const RadialVelErrorEvaluator::DopplerInType &doppler_meas);

}  // namespace se2
}  // namespace steam