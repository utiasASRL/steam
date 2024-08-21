#include "steam/evaluable/p2p/p2plane_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto P2PlaneErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                                   const Eigen::Vector3d &reference,
                                   const Eigen::Vector3d &query,
                                   const Eigen::Vector3d &normal) -> Ptr {
  return std::make_shared<P2PlaneErrorEvaluator>(T_rq, reference, query, normal);
}

P2PlaneErrorEvaluator::P2PlaneErrorEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                                     const Eigen::Vector3d &reference,
                                     const Eigen::Vector3d &query,
                                     const Eigen::Vector3d &normal)
    : T_rq_(T_rq), reference_(reference), query_(query), normal_(normal){
}

bool P2PlaneErrorEvaluator::active() const { return T_rq_->active(); }

void P2PlaneErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_rq_->getRelatedVarKeys(keys);
}

auto P2PlaneErrorEvaluator::value() const -> OutType {
  const auto T_rq = T_rq_->value().matrix();
  return normal_.transpose() * (reference_ - T_rq.block<3, 3>(0, 0) * query_ - T_rq.block<3, 1>(0, 3));
}

auto P2PlaneErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_rq_->forward();
  const auto T_rq = child->value().matrix();
  OutType error = normal_.transpose() * (reference_ - T_rq.block<3, 3>(0, 0) * query_ - T_rq.block<3, 1>(0, 3));
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child);
  return node;
}

void P2PlaneErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_rq_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));

    const auto T_rq = child->value().matrix();
    Eigen::Matrix<double, 3, 1> Tq = T_rq.block<3, 3>(0, 0) * query_ + T_rq.block<3, 1>(0, 3);
    Eigen::Matrix<double, 1, 6> new_lhs = -lhs * normal_.transpose() * lgmath::se3::point2fs(Tq).block<3, 6>(0, 0);

    T_rq_->backward(new_lhs, child, jacs);
  }
}

Eigen::Matrix<double, 1, 6> P2PlaneErrorEvaluator::getJacobianPose() const {
  const auto T_rq = T_rq_->value().matrix();
  Eigen::Matrix<double, 3, 1> Tq = T_rq.block<3, 3>(0, 0) * query_ + T_rq.block<3, 1>(0, 3);
  Eigen::Matrix<double, 1, 6> jac = -normal_.transpose() * lgmath::se3::point2fs(Tq).block<3, 6>(0, 0);
  return jac;
}

P2PlaneErrorEvaluator::Ptr p2planeError(
    const Evaluable<P2PlaneErrorEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query, const Eigen::Vector3d &normal) {
  return P2PlaneErrorEvaluator::MakeShared(T_rq, reference, query, normal);
}

}  // namespace p2p
}  // namespace steam