#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace traj {
namespace const_vel {

class JVelocityEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<JVelocityEvaluator>;
  using ConstPtr = std::shared_ptr<const JVelocityEvaluator>;

  using XiInType = Eigen::Matrix<double, 6, 1>;
  using VelInType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Evaluable<XiInType>::ConstPtr& xi,
                        const Evaluable<VelInType>::ConstPtr& velocity);
  JVelocityEvaluator(const Evaluable<XiInType>::ConstPtr& xi,
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

JVelocityEvaluator::Ptr j_velocity(
    const Evaluable<JVelocityEvaluator::XiInType>::ConstPtr& xi,
    const Evaluable<JVelocityEvaluator::VelInType>::ConstPtr& velocity);

}  // namespace const_vel
}  // namespace traj
}  // namespace steam