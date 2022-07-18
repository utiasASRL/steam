#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class RadialVelErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<RadialVelErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const RadialVelErrorEvaluator>;

  using InType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr &w_iv_inv,
                        const Eigen::Vector3d &pv, const double &r);
  RadialVelErrorEvaluator(const Evaluable<InType>::ConstPtr &w_iv_inv,
                          const Eigen::Vector3d &pv, const double &r);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

 private:
  // evaluable
  const Evaluable<InType>::ConstPtr w_iv_inv_;
  const Eigen::Vector3d pv_;
  const Eigen::Matrix<double, 1, 1> r_;
  Eigen::Matrix<double, 3, 4> D_ = Eigen::Matrix<double, 3, 4>::Zero();
};

RadialVelErrorEvaluator::Ptr radialVelError(
    const Evaluable<RadialVelErrorEvaluator::InType>::ConstPtr &w_iv_inv,
    const Eigen::Vector3d &pv, const double &r);

}  // namespace p2p
}  // namespace steam