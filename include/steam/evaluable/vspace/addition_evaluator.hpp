#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int DIM = Eigen::Dynamic>
class AdditionEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<AdditionEvaluator>;
  using ConstPtr = std::shared_ptr<const AdditionEvaluator>;

  using InType = Eigen::Matrix<double, DIM, 1>;
  using OutType = Eigen::Matrix<double, DIM, 1>;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v1,
                        const typename Evaluable<InType>::ConstPtr& v2);
  AdditionEvaluator(const typename Evaluable<InType>::ConstPtr& v1,
                    const typename Evaluable<InType>::ConstPtr& v2);

  bool active() const override;

  OutType value() const override;
  typename Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs,
                const typename Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const typename Evaluable<InType>::ConstPtr v1_;
  const typename Evaluable<InType>::ConstPtr v2_;
};

// clang-format off
template <int DIM>
typename AdditionEvaluator<DIM>::Ptr add(
    const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v1,
    const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v2);
// clang-format on

}  // namespace vspace
}  // namespace steam

namespace steam {
namespace vspace {

template <int DIM>
auto AdditionEvaluator<DIM>::MakeShared(
    const typename Evaluable<InType>::ConstPtr& v1,
    const typename Evaluable<InType>::ConstPtr& v2) -> Ptr {
  return std::make_shared<AdditionEvaluator>(v1, v2);
}

template <int DIM>
AdditionEvaluator<DIM>::AdditionEvaluator(
    const typename Evaluable<InType>::ConstPtr& v1,
    const typename Evaluable<InType>::ConstPtr& v2)
    : v1_(v1), v2_(v2) {}

template <int DIM>
bool AdditionEvaluator<DIM>::active() const {
  return v1_->active() || v2_->active();
}

template <int DIM>
auto AdditionEvaluator<DIM>::value() const -> OutType {
  return v1_->value() + v2_->value();
}

template <int DIM>
auto AdditionEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
  const auto child1 = v1_->forward();
  const auto child2 = v2_->forward();
  const auto value = child1->value() + child2->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child1);
  node->addChild(child2);
  return node;
}

template <int DIM>
void AdditionEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                      const typename Node<OutType>::Ptr& node,
                                      Jacobians& jacs) const {
  const auto child1 = std::static_pointer_cast<Node<InType>>(node->at(0));
  const auto child2 = std::static_pointer_cast<Node<InType>>(node->at(1));

  if (v1_->active()) {
    v1_->backward(lhs, child1, jacs);
  }

  if (v2_->active()) {
    v2_->backward(lhs, child2, jacs);
  }
}

// clang-format off
template <int DIM>
typename AdditionEvaluator<DIM>::Ptr add(
    const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v1,
    const typename Evaluable<typename AdditionEvaluator<DIM>::InType>::ConstPtr& v2) {
  return AdditionEvaluator<DIM>::MakeShared(v1, v2);
}
// clang-format on

}  // namespace vspace
}  // namespace steam