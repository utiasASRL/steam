#include "steam/evaluable/se3/se3_error_global_perturb_evaluator.hpp"

namespace steam {
namespace se3 {

auto SE3ErrorGlobalPerturbEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_ab,
                                   const InType &T_ab_meas) -> Ptr {
  return std::make_shared<SE3ErrorGlobalPerturbEvaluator>(T_ab, T_ab_meas);
}

SE3ErrorGlobalPerturbEvaluator::SE3ErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr &T_ab,
                                     const InType &T_ab_meas)
    : T_ab_(T_ab), T_ab_meas_(T_ab_meas) {}

bool SE3ErrorGlobalPerturbEvaluator::active() const { return T_ab_->active(); }

void SE3ErrorGlobalPerturbEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_ab_->getRelatedVarKeys(keys);
}

auto SE3ErrorGlobalPerturbEvaluator::value() const -> OutType {
  Eigen::Matrix<double, 6, 1> out = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::Matrix4d T_ab = T_ab_->value().matrix();
  const Eigen::Matrix4d T_ab_meas = T_ab_meas_.matrix();
  out.block<3, 1>(0, 0) = T_ab.block<3, 1>(0, 3)  - T_ab_meas.block<3, 1>(0, 3);
  out.block<3, 1>(3, 0) = lgmath::so3::rot2vec(T_ab_meas.block<3, 3>(0, 0).transpose() * T_ab.block<3, 3>(0, 0));
  return out;
}

auto SE3ErrorGlobalPerturbEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_ab_->forward();
  const Eigen::Matrix4d T_ab = child->value().matrix();
  const Eigen::Matrix4d T_ab_meas = T_ab_meas_.matrix();
  Eigen::Matrix<double, 6, 1> value = Eigen::Matrix<double, 6, 1>::Zero();
  value.block<3, 1>(0, 0) = T_ab.block<3, 1>(0, 3)  - T_ab_meas.block<3, 1>(0, 3);
  value.block<3, 1>(3, 0) = lgmath::so3::rot2vec(T_ab_meas.block<3, 3>(0, 0).transpose() * T_ab.block<3, 3>(0, 0));
  // const auto value = (T_ab_meas_ * child->value().inverse()).vec();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void SE3ErrorGlobalPerturbEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_ab_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    Eigen::Vector3d phi = node->value().block<3, 1>(3, 0);
    Eigen::Matrix<double, 6, 6> jac = Eigen::Matrix<double, 6, 6>::Zero();
    const Eigen::Matrix4d T_ab = child->value().matrix();
    jac.block<3, 3>(0, 0) = T_ab.block<3, 3>(0, 0);
    jac.block<3, 3>(3, 3) = lgmath::so3::vec2jacinv(-phi);
    T_ab_->backward(lhs * jac, child, jacs);
  }
}

SE3ErrorGlobalPerturbEvaluator::Ptr se3_global_perturb_error(
    const Evaluable<SE3ErrorGlobalPerturbEvaluator::InType>::ConstPtr &T_ab,
    const SE3ErrorGlobalPerturbEvaluator::InType &T_ab_meas) {
  return SE3ErrorGlobalPerturbEvaluator::MakeShared(T_ab, T_ab_meas);
}

}  // namespace se3
}  // namespace steam