#include "steam/trajectory/const_vel/evaluable/merge_evaluator.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto MergeEvaluator::MakeShared(
    const Evaluable<PoseInType>::ConstPtr& pose_error,
    const Evaluable<VeloInType>::ConstPtr& velo_error) -> Ptr {
  return std::make_shared<MergeEvaluator>(pose_error, velo_error);
}

MergeEvaluator::MergeEvaluator(
    const Evaluable<PoseInType>::ConstPtr& pose_error,
    const Evaluable<VeloInType>::ConstPtr& velo_error)
    : pose_error_(pose_error), velo_error_(velo_error) {}

bool MergeEvaluator::active() const {
  return pose_error_->active() || velo_error_->active();
}

auto MergeEvaluator::value() const -> OutType {
  OutType value = OutType::Zero();
  value.topRows(6) = pose_error_->value();
  value.bottomRows(6) = velo_error_->value();
  return value;
}

auto MergeEvaluator::forward() const -> Node<OutType>::Ptr {
  //
  const auto child1 = pose_error_->forward();
  const auto child2 = velo_error_->forward();

  //
  OutType value = OutType::Zero();
  value.topRows(6) = child1->value();
  value.bottomRows(6) = child2->value();

  //
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void MergeEvaluator::backward(const Eigen::MatrixXd& lhs,
                              const Node<OutType>::Ptr& node,
                              Jacobians& jacs) const {
  if (pose_error_->active()) {
    const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
    pose_error_->backward(lhs.leftCols(6), child1, jacs);
  }

  if (velo_error_->active()) {
    const auto child2 = std::static_pointer_cast<Node<VeloInType>>(node->at(1));
    velo_error_->backward(lhs.rightCols(6), child2, jacs);
  }
}

MergeEvaluator::Ptr merge(
    const Evaluable<MergeEvaluator::PoseInType>::ConstPtr& pose_error,
    const Evaluable<MergeEvaluator::VeloInType>::ConstPtr& velo_error) {
  return MergeEvaluator::MakeShared(pose_error, velo_error);
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam