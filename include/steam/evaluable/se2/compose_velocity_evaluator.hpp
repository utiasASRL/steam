#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se2 {

class ComposeVelocityEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<ComposeVelocityEvaluator>;
  using ConstPtr = std::shared_ptr<const ComposeVelocityEvaluator>;

  using PoseInType = lgmath::se2::Transformation;
  using VelInType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr& transform,
                        const Evaluable<VelInType>::ConstPtr& velocity);
  ComposeVelocityEvaluator(const Evaluable<PoseInType>::ConstPtr& transform,
                           const Evaluable<VelInType>::ConstPtr& velocity);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<PoseInType>::ConstPtr transform_;
  const Evaluable<VelInType>::ConstPtr velocity_;
};

ComposeVelocityEvaluator::Ptr compose_velocity(
    const Evaluable<ComposeVelocityEvaluator::PoseInType>::ConstPtr& transform,
    const Evaluable<ComposeVelocityEvaluator::VelInType>::ConstPtr& velocity);

}  // namespace se2
}  // namespace steam