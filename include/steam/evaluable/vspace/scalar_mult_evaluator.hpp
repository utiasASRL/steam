#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int DIM = Eigen::Dynamic>
class ScalarMultEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<ScalarMultEvaluator>;
  using ConstPtr = std::shared_ptr<const ScalarMultEvaluator>;

  using InType = Eigen::Matrix<double, DIM, 1>;
  using OutType = Eigen::Matrix<double, DIM, 1>;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v,
                        const double& s);
  ScalarMultEvaluator(const typename Evaluable<InType>::ConstPtr& v,
                      const double& s);

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
  const double s_;
};

// clang-format off
template <int DIM>
typename ScalarMultEvaluator<DIM>::Ptr smult(
    const typename Evaluable<typename ScalarMultEvaluator<DIM>::InType>::ConstPtr& v,
    const double& s);
// clang-format on

}  // namespace vspace
}  // namespace steam

namespace steam {
namespace vspace {

template <int DIM>
auto ScalarMultEvaluator<DIM>::MakeShared(
    const typename Evaluable<InType>::ConstPtr& v, const double& s) -> Ptr {
  return std::make_shared<ScalarMultEvaluator>(v, s);
}

template <int DIM>
ScalarMultEvaluator<DIM>::ScalarMultEvaluator(
    const typename Evaluable<InType>::ConstPtr& v, const double& s)
    : v_(v), s_(s) {}

template <int DIM>
bool ScalarMultEvaluator<DIM>::active() const {
  return v_->active();
}

template <int DIM>
void ScalarMultEvaluator<DIM>::getRelatedVarKeys(KeySet& keys) const {
  v_->getRelatedVarKeys(keys);
}

template <int DIM>
auto ScalarMultEvaluator<DIM>::value() const -> OutType {
  return s_ * v_->value();
}

template <int DIM>
auto ScalarMultEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
  const auto child = v_->forward();
  const auto value = s_ * child->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

template <int DIM>
void ScalarMultEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                        const typename Node<OutType>::Ptr& node,
                                        Jacobians& jacs) const {
  const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
  if (v_->active()) {
    v_->backward(s_ * lhs, child, jacs);
  }
}

// clang-format off
template <int DIM>
typename ScalarMultEvaluator<DIM>::Ptr smult(
    const typename Evaluable<typename ScalarMultEvaluator<DIM>::InType>::ConstPtr& v,
    const double& s) {
  return ScalarMultEvaluator<DIM>::MakeShared(v, s);
}
// clang-format on

}  // namespace vspace
}  // namespace steam