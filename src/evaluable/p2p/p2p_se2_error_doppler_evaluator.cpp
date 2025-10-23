#include "steam/evaluable/p2p/p2p_se2_error_doppler_evaluator.hpp"
namespace steam {
namespace p2p {

auto P2PSE2ErrorDopplerEvaluator::MakeShared(
    const Evaluable<PoseInType>::ConstPtr &T_rq,
    const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector2d &reference, const Eigen::Vector2d &query,
    const float beta, const bool rm_ori) -> Ptr {
  return std::make_shared<P2PSE2ErrorDopplerEvaluator>(T_rq, w_r_q_in_q, reference,
                                                    query, beta, rm_ori);
}

P2PSE2ErrorDopplerEvaluator::P2PSE2ErrorDopplerEvaluator(
    const Evaluable<PoseInType>::ConstPtr &T_rq,
    const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector2d &reference, const Eigen::Vector2d &query,
    const float beta, const bool rm_ori)
    : T_rq_(T_rq), w_r_q_in_q_(w_r_q_in_q), beta_(beta), rm_ori_(rm_ori) {
  D_.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity();
  reference_.block<2, 1>(0, 0) = reference;
  query_.block<2, 1>(0, 0) = query;
}

bool P2PSE2ErrorDopplerEvaluator::active() const {
  return T_rq_->active() || w_r_q_in_q_->active();
}

void P2PSE2ErrorDopplerEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_rq_->getRelatedVarKeys(keys);
  w_r_q_in_q_->getRelatedVarKeys(keys);
}

auto P2PSE2ErrorDopplerEvaluator::value() const -> OutType {
  // clang-format off
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);
  const auto delta_q =
    D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se2::point2fs(D_ * query_, 1.0) * w_r_q_in_q_->value();
  return D_ * (reference_ - T_rq_->value() * (query_ - delta_q));
  // clang-format on
}

auto P2PSE2ErrorDopplerEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child1 = T_rq_->forward();
  const auto child2 = w_r_q_in_q_->forward();

  const auto T_rq = child1->value();
  const auto w_r_q_in_q = child2->value();

  // clang-format off
  const auto abar = D_ * query_ / std::sqrt(query_.transpose() * D_.transpose() * D_ * query_);
  const auto delta_q =
    D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se2::point2fs(D_ * query_, 1.0) * w_r_q_in_q;
  OutType error = D_ * (reference_ - T_rq * (query_ - delta_q));
  // clang-format on

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

void P2PSE2ErrorDopplerEvaluator::backward(const Eigen::MatrixXd &lhs,
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
      D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se2::point2fs(D_ * query_, 1.0) * w_r_q_in_q;
    Eigen::Matrix<double, 2, 1> Tq = (T_rq * (query_ - delta_q)).block<2, 1>(0, 0);
    Eigen::Matrix<double, 2, 3> new_lhs = -D_ * lgmath::se2::point2fs(Tq);
    if (rm_ori_) new_lhs.block<2, 1>(0, 2) = Eigen::Vector2d::Zero();
    T_rq_->backward(lhs * new_lhs, child1, jacs);
  }

  if (w_r_q_in_q_->active()) {
    Eigen::Matrix<double, 2, 3> new_lhs =
      D_ * T_rq.matrix() * D_.transpose() * beta_ * abar * abar.transpose() * D_ * lgmath::se2::point2fs(D_ * query_, 1.0);
    // Zero out orientation contributions since there's no dependency and minor numerical issues can occur
    new_lhs.block<2, 1>(0, 2) = Eigen::Vector2d::Zero();
    w_r_q_in_q_->backward(lhs * new_lhs, child2, jacs);
  }
  // clang-format on
}

P2PSE2ErrorDopplerEvaluator::Ptr p2pSE2ErrorDoppler(
    const Evaluable<P2PSE2ErrorDopplerEvaluator::PoseInType>::ConstPtr &T_rq,
    const Evaluable<P2PSE2ErrorDopplerEvaluator::VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector2d &reference, const Eigen::Vector2d &query,
    const float beta, const bool rm_ori) {
  return P2PSE2ErrorDopplerEvaluator::MakeShared(T_rq, w_r_q_in_q, reference,
                                              query, beta, rm_ori);
}

}  // namespace p2p
}  // namespace steam