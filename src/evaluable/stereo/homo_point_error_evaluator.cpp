#include "steam/evaluable/stereo/homo_point_error_evaluator.hpp"

namespace steam {
namespace stereo {

auto HomoPointErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& pt,
                                         const InType& meas_pt) -> Ptr {
  return std::make_shared<HomoPointErrorEvaluator>(pt, meas_pt);
}

HomoPointErrorEvaluator::HomoPointErrorEvaluator(
    const Evaluable<InType>::ConstPtr& pt, const InType& meas_pt)
    : pt_(pt), meas_pt_(meas_pt) {
  D_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
}

bool HomoPointErrorEvaluator::active() const { return pt_->active(); }

auto HomoPointErrorEvaluator::value() const -> OutType {
  return D_ * (meas_pt_ - pt_->value());
}

auto HomoPointErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = pt_->forward();
  const auto value = D_ * (meas_pt_ - child->value());
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void HomoPointErrorEvaluator::backward(const Eigen::MatrixXd& lhs,
                                       const Node<OutType>::Ptr& node,
                                       Jacobians& jacs) const {
  if (pt_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    Eigen::MatrixXd new_lhs = -lhs * D_;
    pt_->backward(new_lhs, child, jacs);
  }
}

HomoPointErrorEvaluator::Ptr homo_point_error(
    const Evaluable<HomoPointErrorEvaluator::InType>::ConstPtr& pt,
    const HomoPointErrorEvaluator::InType& meas_pt) {
  return HomoPointErrorEvaluator::MakeShared(pt, meas_pt);
}

}  // namespace stereo
}  // namespace steam
