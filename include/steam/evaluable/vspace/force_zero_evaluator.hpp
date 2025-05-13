#pragma once

#include <Eigen/Core>
#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int IN_DIM, int OUT_DIM>
class ForceZeroEvaluator : public Evaluable<Eigen::Matrix<double, OUT_DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<ForceZeroEvaluator>;
  using ConstPtr = std::shared_ptr<const ForceZeroEvaluator>;

  using InType = Eigen::Matrix<double, IN_DIM, 1>;
  using OutType = Eigen::Matrix<double, OUT_DIM, 1>;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v,
                        const Eigen::Matrix<double, OUT_DIM, IN_DIM>& D);
  ForceZeroEvaluator(const typename Evaluable<InType>::ConstPtr& v,
                    const Eigen::Matrix<double, OUT_DIM, IN_DIM>& D);

  bool active() const override;
  using KeySet = typename Evaluable<OutType>::KeySet;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  typename Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs,
                const typename Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const typename Evaluable<InType>::ConstPtr v_;
  Eigen::Matrix<double, OUT_DIM, IN_DIM> D_;
};

// Factory function
template <int IN_DIM, int OUT_DIM>
typename ForceZeroEvaluator<IN_DIM, OUT_DIM>::Ptr force_zero(
    const typename Evaluable<Eigen::Matrix<double, IN_DIM, 1>>::ConstPtr& v,
    const Eigen::Matrix<double, OUT_DIM, IN_DIM>& D) {
  return ForceZeroEvaluator<IN_DIM, OUT_DIM>::MakeShared(v, D);
}

// Implementations

template <int IN_DIM, int OUT_DIM>
typename ForceZeroEvaluator<IN_DIM, OUT_DIM>::Ptr
ForceZeroEvaluator<IN_DIM, OUT_DIM>::MakeShared(
    const typename Evaluable<InType>::ConstPtr& v,
    const Eigen::Matrix<double, OUT_DIM, IN_DIM>& D) {
  return std::make_shared<ForceZeroEvaluator>(v, D);
}

template <int IN_DIM, int OUT_DIM>
ForceZeroEvaluator<IN_DIM, OUT_DIM>::ForceZeroEvaluator(
    const typename Evaluable<InType>::ConstPtr& v,
    const Eigen::Matrix<double, OUT_DIM, IN_DIM>& D)
    : v_(v), D_(D) {}

template <int IN_DIM, int OUT_DIM>
bool ForceZeroEvaluator<IN_DIM, OUT_DIM>::active() const {
  return v_->active();
}

template <int IN_DIM, int OUT_DIM>
void ForceZeroEvaluator<IN_DIM, OUT_DIM>::getRelatedVarKeys(KeySet& keys) const {
  v_->getRelatedVarKeys(keys);
}

template <int IN_DIM, int OUT_DIM>
typename ForceZeroEvaluator<IN_DIM, OUT_DIM>::OutType
ForceZeroEvaluator<IN_DIM, OUT_DIM>::value() const {
  return D_ * v_->value();
}

template <int IN_DIM, int OUT_DIM>
typename Node<typename ForceZeroEvaluator<IN_DIM, OUT_DIM>::OutType>::Ptr
ForceZeroEvaluator<IN_DIM, OUT_DIM>::forward() const {
  const auto child = v_->forward();
  const auto value = D_ * child->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

template <int IN_DIM, int OUT_DIM>
void ForceZeroEvaluator<IN_DIM, OUT_DIM>::backward(
    const Eigen::MatrixXd& lhs,
    const typename Node<OutType>::Ptr& node,
    Jacobians& jacs) const {
  if (v_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    v_->backward(lhs * D_, child, jacs);
  }
}

}  // namespace vspace
}  // namespace steam
