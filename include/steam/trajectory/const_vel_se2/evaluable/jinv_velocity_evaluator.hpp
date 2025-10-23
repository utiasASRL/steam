#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace traj {
namespace const_vel_se2 {

class JinvVelocityEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<JinvVelocityEvaluator>;
  using ConstPtr = std::shared_ptr<const JinvVelocityEvaluator>;

  using XiInType = Eigen::Matrix<double, 3, 1>;
  using VelInType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<XiInType>::ConstPtr& xi,
                        const Evaluable<VelInType>::ConstPtr& velocity);
  JinvVelocityEvaluator(const Evaluable<XiInType>::ConstPtr& xi,
                        const Evaluable<VelInType>::ConstPtr& velocity);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<XiInType>::ConstPtr xi_;
  const Evaluable<VelInType>::ConstPtr velocity_;
};

JinvVelocityEvaluator::Ptr jinv_velocity(
    const Evaluable<JinvVelocityEvaluator::XiInType>::ConstPtr& xi,
    const Evaluable<JinvVelocityEvaluator::VelInType>::ConstPtr& velocity);

}  // namespace const_vel_se2
}  // namespace traj
}  // namespace steam