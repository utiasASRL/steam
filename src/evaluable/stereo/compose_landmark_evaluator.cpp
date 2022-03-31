#include "steam/evaluable/stereo/compose_landmark_evaluator.hpp"

namespace steam {
namespace stereo {

auto ComposeLandmarkEvaluator::MakeShared(
    const Evaluable<PoseInType>::ConstPtr& transform,
    const Evaluable<LmInType>::ConstPtr& landmark) -> Ptr {
  return std::make_shared<ComposeLandmarkEvaluator>(transform, landmark);
}

ComposeLandmarkEvaluator::ComposeLandmarkEvaluator(
    const Evaluable<PoseInType>::ConstPtr& transform,
    const Evaluable<LmInType>::ConstPtr& landmark)
    : transform_(transform), landmark_(landmark) {}

bool ComposeLandmarkEvaluator::active() const {
  return transform_->active() || landmark_->active();
}

auto ComposeLandmarkEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = transform_->forward();
  const auto child2 = landmark_->forward();
  const auto value = child1->value() * child2->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);
  return node;
}

void ComposeLandmarkEvaluator::backward(const Eigen::MatrixXd& lhs,
                                        const Node<OutType>::Ptr& node,
                                        Jacobians& jacs) const {
  const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<LmInType>>(node->at(1));

  if (transform_->active()) {
    const auto& homogeneous = node->value();
    Eigen::MatrixXd new_lhs =
        lhs * lgmath::se3::point2fs(homogeneous.head<3>(), homogeneous[3]);
    transform_->backward(new_lhs, child1, jacs);
  }

  if (landmark_->active()) {
    // Construct Jacobian
    Eigen::Matrix<double, 4, 6> land_jac;
    land_jac.block<4, 3>(0, 0) = child1->value().matrix().block<4, 3>(0, 0);
    landmark_->backward(lhs * land_jac, child2, jacs);
  }
}

ComposeLandmarkEvaluator::Ptr compose(
    const Evaluable<ComposeLandmarkEvaluator::PoseInType>::ConstPtr& transform,
    const Evaluable<ComposeLandmarkEvaluator::LmInType>::ConstPtr& landmark) {
  return ComposeLandmarkEvaluator::MakeShared(transform, landmark);
}

}  // namespace stereo
}  // namespace steam
