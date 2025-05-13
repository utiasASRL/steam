#include "steam/evaluable/se3/se3_3d_error_evaluator.hpp"

namespace steam {
namespace se3 {

auto SE33DErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr &T_ab) -> Ptr {
  return std::make_shared<SE33DErrorEvaluator>(T_ab);
}

SE33DErrorEvaluator::SE33DErrorEvaluator(const Evaluable<InType>::ConstPtr &T_ab)
    : T_ab_(T_ab) {
      D_ << 0, 0, 1, 0, 0, 0, 
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0;
    }

bool SE33DErrorEvaluator::active() const { return T_ab_->active(); }

void SE33DErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
  T_ab_->getRelatedVarKeys(keys);
}

auto SE33DErrorEvaluator::value() const -> OutType {
  return D_ * T_ab_->value().vec();
}

auto SE33DErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = T_ab_->forward();
  const auto value = D_ * child->value().vec();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void SE33DErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                 const Node<OutType>::Ptr &node,
                                 Jacobians &jacs) const {
  if (T_ab_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    T_ab_->backward(lhs * D_, child, jacs);
  }
}

SE33DErrorEvaluator::Ptr se3_3d_error(
    const Evaluable<SE33DErrorEvaluator::InType>::ConstPtr &T_ab) {
  return SE33DErrorEvaluator::MakeShared(T_ab);
}

}  // namespace se3
}  // namespace steam