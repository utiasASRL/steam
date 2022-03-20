#include "steam/evaluable/p2p/p2p_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto P2PErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                                   const Eigen::Vector3d &reference,
                                   const Eigen::Vector3d &query) -> Ptr {
  return std::make_shared<P2PErrorEvaluator>(T_rq, reference, query);
}

P2PErrorEvaluator::P2PErrorEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                                     const Eigen::Vector3d &reference,
                                     const Eigen::Vector3d &query)
    : T_rq_(T_rq) {
  D_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  reference_.block<3, 1>(0, 0) = reference;
  query_.block<3, 1>(0, 0) = query;
}

bool P2PErrorEvaluator::active() const { return T_rq_->active(); }

auto P2PErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_rq_->forward();
  const auto T_rq = child->value();
  OutType error = D_ * (reference_ - T_rq * query_);
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child);
  return node;
}

void P2PErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_rq_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));

    const auto T_rq = child->value();
    Eigen::Matrix<double, 3, 1> Tq = (T_rq * query_).block<3, 1>(0, 0);
    Eigen::Matrix<double, 3, 6> new_lhs = -lhs * D_ * lgmath::se3::point2fs(Tq);

    T_rq_->backward(new_lhs, child, jacs);
  }
}

P2PErrorEvaluator::Ptr p2pError(
    const Evaluable<P2PErrorEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query) {
  return P2PErrorEvaluator::MakeShared(T_rq, reference, query);
}

}  // namespace p2p
}  // namespace steam