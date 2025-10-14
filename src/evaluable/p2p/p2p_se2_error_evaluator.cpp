#include "steam/evaluable/p2p/p2p_se2_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto P2PSE2ErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                                   const Eigen::Vector3d &reference,
                                   const Eigen::Vector3d &query,
                                   const bool rm_ori) -> Ptr {
  return std::make_shared<P2PSE2ErrorEvaluator>(T_rq, reference, query, rm_ori);
}

P2PSE2ErrorEvaluator::P2PSE2ErrorEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                                     const Eigen::Vector3d &reference,
                                     const Eigen::Vector3d &query,
                                     const bool rm_ori)
    : T_rq_(T_rq), rm_ori_(rm_ori) {
  D_.block<2, 2>(0, 0) = Eigen::Matrix3d::Identity();
  reference_.block<2, 1>(0, 0) = reference;
  query_.block<2, 1>(0, 0) = query;
}

bool P2PSE2ErrorEvaluator::active() const { return T_rq_->active(); }

void P2PSE2ErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_rq_->getRelatedVarKeys(keys);
}

auto P2PSE2ErrorEvaluator::value() const -> OutType {
  return D_ * (reference_ - T_rq_->value() * query_);
}

auto P2PSE2ErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_rq_->forward();
  const auto T_rq = child->value();
  OutType error = D_ * (reference_ - T_rq * query_);
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child);
  return node;
}

void P2PSE2ErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_rq_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));

    const auto T_rq = child->value();
    Eigen::Matrix<double, 2, 1> Tq = (T_rq * query_).block<2, 1>(0, 0);
    Eigen::Matrix<double, 2, 3> new_lhs = - D_ * lgmath::se2::point2fs(Tq);
    if (rm_ori_) new_lhs.block<2, 1>(0, 2) = Eigen::Vector2d::Zero();

    T_rq_->backward(lhs * new_lhs, child, jacs);
  }
}

Eigen::Matrix<double, 2, 3> P2PSE2ErrorEvaluator::getJacobianPose() const {
  const auto T_rq = T_rq_->value();
  Eigen::Matrix<double, 2, 1> Tq = (T_rq * query_).block<2, 1>(0, 0);
  Eigen::Matrix<double, 2, 3> jac = -D_ * lgmath::se2::point2fs(Tq);
  return jac;
}

P2PSE2ErrorEvaluator::Ptr p2pSE2Error(
    const Evaluable<P2PSE2ErrorEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query, const bool rm_ori) {
  return P2PSE2ErrorEvaluator::MakeShared(T_rq, reference, query, rm_ori);
}

}  // namespace p2p
}  // namespace steam