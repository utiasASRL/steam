#include "steam/evaluable/p2p/p2p_error_doppler_evaluator.hpp"

namespace steam {
namespace p2p {

auto P2PErrorWithDopplerCompensationEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                                   const Eigen::Vector3d &reference,
                                   const Eigen::Vector3d &query,
                                   const float beta,
                                   const Evaluable<VType>::ConstPtr &w_r_q_in_q) -> Ptr {
  return std::make_shared<P2PErrorWithDopplerCompensationEvaluator>(T_rq, reference, query, beta, w_r_q_in_q);
}

P2PErrorWithDopplerCompensationEvaluator::P2PErrorWithDopplerCompensationEvaluator(
                                   const Evaluable<InType>::ConstPtr &T_rq,
                                   const Eigen::Vector3d &reference,
                                   const Eigen::Vector3d &query,
                                   const float beta,
                                   const Evaluable<VType>::ConstPtr &w_r_q_in_q)
    : T_rq_(T_rq), beta_(beta), w_r_q_in_q_(w_r_q_in_q) {
  D_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  reference_.block<3, 1>(0, 0) = reference;
  query_.block<3, 1>(0, 0) = query;
}

bool P2PErrorWithDopplerCompensationEvaluator::active() const { return T_rq_->active(); }

auto P2PErrorWithDopplerCompensationEvaluator::value() const -> OutType {
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);
  const auto delta_q = D_.transpose() * beta_ * abar * abar.transpose() *
    D_ * lgmath::se3::point2fs(D_ * query_, 1.0) * w_r_q_in_q_->value();
  return D_ * (reference_ - T_rq_->value() * (query_ - delta_q));
}

auto P2PErrorWithDopplerCompensationEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = T_rq_->forward();
  const auto T_rq = child1->value();
  const auto child2 = w_r_q_in_q_->forward();
  const auto w_r_q_in_q = child2->value();
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);
  const auto delta_q = D_.transpose() * beta_ * abar * abar.transpose() *
    D_ * lgmath::se3::point2fs(D_ * query_, 1.0) * w_r_q_in_q;
  OutType error = D_ * (reference_ - T_rq * (query_ - delta_q));
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);
  return node;
}

void P2PErrorWithDopplerCompensationEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<VType>>(node->at(1));
  const auto T_rq = child1->value();
  const auto w_r_q_in_q = child2->value();
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);

  if (T_rq_->active()) {
    const auto delta_q = D_.transpose() * beta_ * abar * abar.transpose() *
      D_ * lgmath::se3::point2fs(D_ * query_, 1.0) * w_r_q_in_q;
    Eigen::Matrix<double, 3, 1> Tq = (T_rq * (query_ - delta_q)).block<3, 1>(0, 0);
    Eigen::Matrix<double, 3, 6> new_lhs = -lhs * D_ * lgmath::se3::point2fs(Tq);
    T_rq_->backward(new_lhs, child1, jacs);
  }

  if (w_r_q_in_q_->active()) {
    Eigen::Matrix<double, 3, 6> new_lhs = lhs * D_ * T_rq.matrix() * D_.transpose() * beta_ * abar *
      abar.transpose() * D_ * lgmath::se3::point2fs(D_ * query_, 1.0);
    w_r_q_in_q_->backward(new_lhs, child2, jacs);
  }
}

P2PErrorWithDopplerCompensationEvaluator::Ptr p2pError(
    const Evaluable<P2PErrorWithDopplerCompensationEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query, const float beta,
    const Evaluable<P2PErrorWithDopplerCompensationEvaluator::VType>::ConstPtr &w_r_q_in_q) {
  return P2PErrorWithDopplerCompensationEvaluator::MakeShared(T_rq, reference, query, beta, w_r_q_in_q);
}

}  // namespace p2p
}  // namespace steam