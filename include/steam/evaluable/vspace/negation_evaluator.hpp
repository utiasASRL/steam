#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int DIM = Eigen::Dynamic>
class NegationEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<NegationEvaluator>;
  using ConstPtr = std::shared_ptr<const NegationEvaluator>;

  using InType = Eigen::Matrix<double, DIM, 1>;
  using OutType = Eigen::Matrix<double, DIM, 1>;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v);
  NegationEvaluator(const typename Evaluable<InType>::ConstPtr& v);

  bool active() const override;

  OutType value() const override;
  typename Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs,
                const typename Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const typename Evaluable<InType>::ConstPtr v_;
};
// clang-format off
template <int DIM>
typename NegationEvaluator<DIM>::Ptr neg(
    const typename Evaluable<typename NegationEvaluator<DIM>::InType>::ConstPtr& v);
// clang-format on
}  // namespace vspace
}  // namespace steam

namespace steam {
namespace vspace {

template <int DIM>
auto NegationEvaluator<DIM>::MakeShared(
    const typename Evaluable<InType>::ConstPtr& v) -> Ptr {
  return std::make_shared<NegationEvaluator>(v);
}

template <int DIM>
NegationEvaluator<DIM>::NegationEvaluator(
    const typename Evaluable<InType>::ConstPtr& v)
    : v_(v) {}

template <int DIM>
bool NegationEvaluator<DIM>::active() const {
  return v_->active();
}

template <int DIM>
auto NegationEvaluator<DIM>::value() const -> OutType {
  return -v_->value();
}

template <int DIM>
auto NegationEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
  const auto child = v_->forward();
  const auto value = -child->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

template <int DIM>
void NegationEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                      const typename Node<OutType>::Ptr& node,
                                      Jacobians& jacs) const {
  const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
  if (v_->active()) {
    v_->backward(-lhs, child, jacs);
  }
}

// clang-format off
template <int DIM>
typename NegationEvaluator<DIM>::Ptr neg(
    const typename Evaluable<typename NegationEvaluator<DIM>::InType>::ConstPtr& v) {
  return NegationEvaluator<DIM>::MakeShared(v);
}
// clang-format on

}  // namespace vspace
}  // namespace steam