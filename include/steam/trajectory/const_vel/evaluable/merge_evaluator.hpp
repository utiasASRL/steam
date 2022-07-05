#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace traj {
namespace const_vel {

class MergeEvaluator : public Evaluable<Eigen::Matrix<double, 12, 1>> {
 public:
  using Ptr = std::shared_ptr<MergeEvaluator>;
  using ConstPtr = std::shared_ptr<const MergeEvaluator>;

  using PoseInType = Eigen::Matrix<double, 6, 1>;
  using VeloInType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 12, 1>;

  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr& pose_error,
                        const Evaluable<VeloInType>::ConstPtr& velo_error);
  MergeEvaluator(const Evaluable<PoseInType>::ConstPtr& pose_error,
                 const Evaluable<VeloInType>::ConstPtr& velo_error);

  bool active() const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<PoseInType>::ConstPtr pose_error_;
  const Evaluable<VeloInType>::ConstPtr velo_error_;
};

MergeEvaluator::Ptr merge(
    const Evaluable<MergeEvaluator::PoseInType>::ConstPtr& pose_error,
    const Evaluable<MergeEvaluator::VeloInType>::ConstPtr& velo_error);

}  // namespace const_vel
}  // namespace traj
}  // namespace steam