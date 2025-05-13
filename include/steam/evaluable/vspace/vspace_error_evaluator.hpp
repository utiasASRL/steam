#pragma once

#include <Eigen/Core>
#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int DIM = Eigen::Dynamic>
class VSpaceErrorEvaluator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<VSpaceErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const VSpaceErrorEvaluator>;

  using InType = Eigen::Matrix<double, DIM, 1>;
  using OutType = Eigen::Matrix<double, DIM, 1>;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v,
                        const InType& v_meas);
  VSpaceErrorEvaluator(const typename Evaluable<InType>::ConstPtr& v,
                       const InType& v_meas);

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
  const InType v_meas_;
};
// clang-format off
template <int DIM>
typename VSpaceErrorEvaluator<DIM>::Ptr vspace_error(
    const typename Evaluable<typename VSpaceErrorEvaluator<DIM>::InType>::ConstPtr& v,
    const typename VSpaceErrorEvaluator<DIM>::InType& v_meas);
// clang-format on
}  // namespace vspace
}  // namespace steam

namespace steam {
namespace vspace {

template <int DIM>
auto VSpaceErrorEvaluator<DIM>::MakeShared(
    const typename Evaluable<InType>::ConstPtr& v, const InType& v_meas)
    -> Ptr {
  return std::make_shared<VSpaceErrorEvaluator>(v, v_meas);
}

template <int DIM>
VSpaceErrorEvaluator<DIM>::VSpaceErrorEvaluator(
    const typename Evaluable<InType>::ConstPtr& v, const InType& v_meas)
    : v_(v), v_meas_(v_meas) {}

template <int DIM>
bool VSpaceErrorEvaluator<DIM>::active() const {
  return v_->active();
}

template <int DIM>
void VSpaceErrorEvaluator<DIM>::getRelatedVarKeys(KeySet& keys) const {
  v_->getRelatedVarKeys(keys);
}

template <int DIM>
auto VSpaceErrorEvaluator<DIM>::value() const -> OutType {
  return v_meas_ - v_->value();
}

template <int DIM>
auto VSpaceErrorEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
  const auto child = v_->forward();
  const auto value = v_meas_ - child->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

template <int DIM>
void VSpaceErrorEvaluator<DIM>::backward(
    const Eigen::MatrixXd& lhs, const typename Node<OutType>::Ptr& node,
    Jacobians& jacs) const {
  if (v_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    v_->backward(-lhs, child, jacs);
  }
}

// clang-format off
template <int DIM>
typename VSpaceErrorEvaluator<DIM>::Ptr vspace_error(
    const typename Evaluable<typename VSpaceErrorEvaluator<DIM>::InType>::ConstPtr& v,
    const typename VSpaceErrorEvaluator<DIM>::InType& v_meas) {
  return VSpaceErrorEvaluator<DIM>::MakeShared(v, v_meas);
}
// clang-format on

}  // namespace vspace
}  // namespace steam