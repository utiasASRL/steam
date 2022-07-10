#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int DIM1, int DIM2>
class MergeEvaluator : public Evaluable<Eigen::Matrix<double, DIM1 + DIM2, 1>> {
 public:
  using Ptr = std::shared_ptr<MergeEvaluator>;
  using ConstPtr = std::shared_ptr<const MergeEvaluator>;

  using In1Type = Eigen::Matrix<double, DIM1, 1>;
  using In2Type = Eigen::Matrix<double, DIM2, 1>;
  using OutType = Eigen::Matrix<double, DIM1 + DIM2, 1>;

  static Ptr MakeShared(const typename Evaluable<In1Type>::ConstPtr& v1,
                        const typename Evaluable<In2Type>::ConstPtr& v2);
  MergeEvaluator(const typename Evaluable<In1Type>::ConstPtr& v1,
                 const typename Evaluable<In2Type>::ConstPtr& v2);

  bool active() const override;

  OutType value() const override;
  typename Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs,
                const typename Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const typename Evaluable<In1Type>::ConstPtr v1_;
  const typename Evaluable<In2Type>::ConstPtr v2_;
};

// clang-format off
template <int DIM1, int DIM2>
typename MergeEvaluator<DIM1, DIM2>::Ptr merge(
    const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In1Type>::ConstPtr& v1,
    const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In2Type>::ConstPtr& v2);
// clang-format on

}  // namespace vspace
}  // namespace steam

namespace steam {
namespace vspace {

template <int DIM1, int DIM2>
auto MergeEvaluator<DIM1, DIM2>::MakeShared(
    const typename Evaluable<In1Type>::ConstPtr& v1,
    const typename Evaluable<In2Type>::ConstPtr& v2) -> Ptr {
  return std::make_shared<MergeEvaluator>(v1, v2);
}

template <int DIM1, int DIM2>
MergeEvaluator<DIM1, DIM2>::MergeEvaluator(
    const typename Evaluable<In1Type>::ConstPtr& v1,
    const typename Evaluable<In2Type>::ConstPtr& v2)
    : v1_(v1), v2_(v2) {}

template <int DIM1, int DIM2>
bool MergeEvaluator<DIM1, DIM2>::active() const {
  return v1_->active() || v2_->active();
}

template <int DIM1, int DIM2>
auto MergeEvaluator<DIM1, DIM2>::value() const -> OutType {
  OutType value = OutType::Zero();
  value.topRows(DIM1) = v1_->value();
  value.bottomRows(DIM2) = v2_->value();
  return value;
}

template <int DIM1, int DIM2>
auto MergeEvaluator<DIM1, DIM2>::forward() const ->
    typename Node<OutType>::Ptr {
  //
  const auto child1 = v1_->forward();
  const auto child2 = v2_->forward();

  //
  OutType value = OutType::Zero();
  value.topRows(DIM1) = child1->value();
  value.bottomRows(DIM2) = child2->value();

  //
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);

  return node;
}

template <int DIM1, int DIM2>
void MergeEvaluator<DIM1, DIM2>::backward(
    const Eigen::MatrixXd& lhs, const typename Node<OutType>::Ptr& node,
    Jacobians& jacs) const {
  if (v1_->active()) {
    const auto child1 = std::static_pointer_cast<Node<In1Type>>(node->at(0));
    v1_->backward(lhs.leftCols(DIM1), child1, jacs);
  }

  if (v2_->active()) {
    const auto child2 = std::static_pointer_cast<Node<In2Type>>(node->at(1));
    v2_->backward(lhs.rightCols(DIM2), child2, jacs);
  }
}

// clang-format off
template <int DIM1, int DIM2>
typename MergeEvaluator<DIM1, DIM2>::Ptr merge(
    const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In1Type>::ConstPtr& v1,
    const typename Evaluable<typename MergeEvaluator<DIM1, DIM2>::In2Type>::ConstPtr& v2) {
  return MergeEvaluator<DIM1, DIM2>::MakeShared(v1, v2);
}
// clang-format on

}  // namespace vspace
}  // namespace steam