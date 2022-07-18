#include "steam/evaluable/p2p/p2p_rv_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto P2PRVErrorEvaluator::MakeShared(const Evaluable<InP2PType>::ConstPtr &p2p,
                                     const Evaluable<InRVType>::ConstPtr &rv)
    -> Ptr {
  return std::make_shared<P2PRVErrorEvaluator>(p2p, rv);
}

P2PRVErrorEvaluator::P2PRVErrorEvaluator(
    const Evaluable<InP2PType>::ConstPtr &p2p,
    const Evaluable<InRVType>::ConstPtr &rv)
    : p2p_(p2p), rv_(rv) {}

bool P2PRVErrorEvaluator::active() const {
  return p2p_->active() || rv_->active();
}

void P2PRVErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  p2p_->getRelatedVarKeys(keys);
  rv_->getRelatedVarKeys(keys);
}

auto P2PRVErrorEvaluator::value() const -> OutType {
  OutType error = OutType::Zero();
  error.block<3, 1>(0, 0) = p2p_->value();
  error.block<1, 1>(3, 0) = rv_->value();
  return error;
}

auto P2PRVErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  OutType error = OutType::Zero();

  const auto p2p_child = p2p_->forward();
  error.block<3, 1>(0, 0) = p2p_child->value();

  const auto rv_child = rv_->forward();
  error.block<1, 1>(3, 0) = rv_child->value();

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(p2p_child);
  node->addChild(rv_child);
  return node;
}

void P2PRVErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                   const Node<OutType>::Ptr &node,
                                   Jacobians &jacs) const {
  if (p2p_->active()) {
    const auto child = std::static_pointer_cast<Node<InP2PType>>(node->at(0));
    p2p_->backward(lhs.block<3, 1>(0, 0), child, jacs);
  }
  if (rv_->active()) {
    const auto child = std::static_pointer_cast<Node<InRVType>>(node->at(1));
    rv_->backward(lhs.block<1, 1>(3, 0), child, jacs);
  }
}

P2PRVErrorEvaluator::Ptr p2prvError(
    const Evaluable<P2PRVErrorEvaluator::InP2PType>::ConstPtr &p2p,
    const Evaluable<P2PRVErrorEvaluator::InRVType>::ConstPtr &rv) {
  return P2PRVErrorEvaluator::MakeShared(p2p, rv);
}

}  // namespace p2p
}  // namespace steam