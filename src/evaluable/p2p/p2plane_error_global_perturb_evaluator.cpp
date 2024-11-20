#include "steam/evaluable/p2p/p2plane_error_global_perturb_evaluator.hpp"

namespace steam {
namespace p2p {

auto P2PlaneErrorGlobalPerturbEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                                   const Eigen::Vector3d &reference,
                                   const Eigen::Vector3d &query,
                                   const Eigen::Vector3d &normal) -> Ptr {
  return std::make_shared<P2PlaneErrorGlobalPerturbEvaluator>(T_rq, reference, query, normal);
}

P2PlaneErrorGlobalPerturbEvaluator::P2PlaneErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                                     const Eigen::Vector3d &reference,
                                     const Eigen::Vector3d &query,
                                     const Eigen::Vector3d &normal)
    : T_rq_(T_rq), reference_(reference), query_(query), normal_(normal){
}

bool P2PlaneErrorGlobalPerturbEvaluator::active() const { return T_rq_->active(); }

void P2PlaneErrorGlobalPerturbEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_rq_->getRelatedVarKeys(keys);
}

auto P2PlaneErrorGlobalPerturbEvaluator::value() const -> OutType {
  const auto T_rq = T_rq_->value().matrix();
  return normal_.transpose() * (reference_ - T_rq.block<3, 3>(0, 0) * query_ - T_rq.block<3, 1>(0, 3));
}

auto P2PlaneErrorGlobalPerturbEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_rq_->forward();
  const auto T_rq = child->value().matrix();
  OutType error = normal_.transpose() * (reference_ - T_rq.block<3, 3>(0, 0) * query_ - T_rq.block<3, 1>(0, 3));
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child);
  return node;
}

void P2PlaneErrorGlobalPerturbEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_rq_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));

    const Eigen::Matrix4d T_rq = child->value().matrix();
    const Eigen::Matrix3d C_rq = T_rq.block<3, 3>(0, 0);
    Eigen::Matrix<double, 1, 6> jac = Eigen::Matrix<double, 1, 6>::Zero();
    // d e / d delta_r
    jac.block<1, 3>(0, 0) = -normal_.transpose() * C_rq;
    // d e / d delta_phi
    jac.block<1, 3>(0, 3) = normal_.transpose() * C_rq * lgmath::so3::hat(query_);

    T_rq_->backward(lhs * jac, child, jacs);
  }
}

Eigen::Matrix<double, 1, 6> P2PlaneErrorGlobalPerturbEvaluator::getJacobianPose() const {
  const auto T_rq = T_rq_->value().matrix();
  const Eigen::Matrix3d C_rq = T_rq.block<3, 3>(0, 0);
  Eigen::Matrix<double, 1, 6> jac = Eigen::Matrix<double, 1, 6>::Zero();
  // d e / d delta_r
  jac.block<1, 3>(0, 0) = -normal_.transpose();
  // d e / d delta_phi
  jac.block<1, 3>(0, 3) = normal_.transpose() * C_rq * lgmath::so3::hat(query_);
  return jac;
}

P2PlaneErrorGlobalPerturbEvaluator::Ptr p2planeGlobalError(
    const Evaluable<P2PlaneErrorGlobalPerturbEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query, const Eigen::Vector3d &normal) {
  return P2PlaneErrorGlobalPerturbEvaluator::MakeShared(T_rq, reference, query, normal);
}

}  // namespace p2p
}  // namespace steam