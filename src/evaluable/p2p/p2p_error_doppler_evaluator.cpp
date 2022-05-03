#include "steam/evaluable/p2p/p2p_error_doppler_evaluator.hpp"

namespace steam {
namespace p2p {

auto P2PErrorDopplerEvaluator::MakeShared(
    const Evaluable<PoseInType>::ConstPtr &T_rq,
    const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query,
    const float beta) -> Ptr {
  return std::make_shared<P2PErrorDopplerEvaluator>(T_rq, w_r_q_in_q, reference,
                                                    query, beta);
}

P2PErrorDopplerEvaluator::P2PErrorDopplerEvaluator(
    const Evaluable<PoseInType>::ConstPtr &T_rq,
    const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query,
    const float beta)
    : T_rq_(T_rq), w_r_q_in_q_(w_r_q_in_q), beta_(beta) {
  D_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  reference_.block<3, 1>(0, 0) = reference;
  query_.block<3, 1>(0, 0) = query;
}

bool P2PErrorDopplerEvaluator::active() const {
  return T_rq_->active() || w_r_q_in_q_->active();
}

auto P2PErrorDopplerEvaluator::value() const -> OutType {
  // clang-format off
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);
  const auto delta_q =
    D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se3::point2fs(D_ * query_, 1.0) * w_r_q_in_q_->value();
  return D_ * (reference_ - T_rq_->value() * (query_ - delta_q));
  // clang-format on
}

auto P2PErrorDopplerEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = T_rq_->forward();
  const auto child2 = w_r_q_in_q_->forward();

  const auto T_rq = child1->value();
  const auto w_r_q_in_q = child2->value();

  // clang-format off
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);
  const auto delta_q =
    D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se3::point2fs(D_ * query_, 1.0) * w_r_q_in_q;
  OutType error = D_ * (reference_ - T_rq * (query_ - delta_q));
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void P2PErrorDopplerEvaluator::backward(const Eigen::MatrixXd &lhs,
                                        const Node<OutType>::Ptr &node,
                                        Jacobians &jacs) const {
  const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<VelInType>>(node->at(1));

  const auto T_rq = child1->value();
  const auto w_r_q_in_q = child2->value();
  // clang-format off
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);

  if (T_rq_->active()) {
    const auto delta_q =
      D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se3::point2fs(D_ * query_, 1.0) * w_r_q_in_q;
    Eigen::Matrix<double, 3, 1> Tq = (T_rq * (query_ - delta_q)).block<3, 1>(0, 0);
    Eigen::Matrix<double, 3, 6> new_lhs = -lhs * D_ * lgmath::se3::point2fs(Tq);
    T_rq_->backward(new_lhs, child1, jacs);
  }

  if (w_r_q_in_q_->active()) {
    Eigen::Matrix<double, 3, 6> new_lhs =
      lhs * D_ * T_rq.matrix() * D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se3::point2fs(D_ * query_, 1.0);
    w_r_q_in_q_->backward(new_lhs, child2, jacs);
  }
  // clang-format on
}

P2PErrorDopplerEvaluator::Ptr p2pErrorDoppler(
    const Evaluable<P2PErrorDopplerEvaluator::PoseInType>::ConstPtr &T_rq,
    const Evaluable<P2PErrorDopplerEvaluator::VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query,
    const float beta) {
  return P2PErrorDopplerEvaluator::MakeShared(T_rq, w_r_q_in_q, reference,
                                              query, beta);
}

}  // namespace p2p
}  // namespace steam