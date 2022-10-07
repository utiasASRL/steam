#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int ROW = Eigen::Dynamic, int COL = ROW>
class MatrixMultEvaluator : public Evaluable<Eigen::Matrix<double, ROW, 1>> {
 public:
  using Ptr = std::shared_ptr<MatrixMultEvaluator>;
  using ConstPtr = std::shared_ptr<const MatrixMultEvaluator>;

  using MatType = Eigen::Matrix<double, ROW, COL>;
  using InType = Eigen::Matrix<double, COL, 1>;
  using OutType = Eigen::Matrix<double, ROW, 1>;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v,
                        const MatType& s);
  MatrixMultEvaluator(const typename Evaluable<InType>::ConstPtr& v,
                      const MatType& s);

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
  const MatType s_;
};

// clang-format off
template <int ROW, int COL = ROW>
typename MatrixMultEvaluator<ROW, COL>::Ptr mmult(
    const typename Evaluable<typename MatrixMultEvaluator<ROW, COL>::InType>::ConstPtr& v,
    const typename MatrixMultEvaluator<ROW, COL>::MatType& s);
// clang-format on

}  // namespace vspace
}  // namespace steam

namespace steam {
namespace vspace {

template <int ROW, int COL>
auto MatrixMultEvaluator<ROW, COL>::MakeShared(
    const typename Evaluable<InType>::ConstPtr& v, const MatType& s) -> Ptr {
  return std::make_shared<MatrixMultEvaluator>(v, s);
}

template <int ROW, int COL>
MatrixMultEvaluator<ROW, COL>::MatrixMultEvaluator(
    const typename Evaluable<InType>::ConstPtr& v, const MatType& s)
    : v_(v), s_(s) {}

template <int ROW, int COL>
bool MatrixMultEvaluator<ROW, COL>::active() const {
  return v_->active();
}

template <int ROW, int COL>
void MatrixMultEvaluator<ROW, COL>::getRelatedVarKeys(KeySet& keys) const {
  v_->getRelatedVarKeys(keys);
}

template <int ROW, int COL>
auto MatrixMultEvaluator<ROW, COL>::value() const -> OutType {
  return s_ * v_->value();
}

template <int ROW, int COL>
auto MatrixMultEvaluator<ROW, COL>::forward() const ->
    typename Node<OutType>::Ptr {
  const auto child = v_->forward();
  const auto value = s_ * child->value();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

template <int ROW, int COL>
void MatrixMultEvaluator<ROW, COL>::backward(
    const Eigen::MatrixXd& lhs, const typename Node<OutType>::Ptr& node,
    Jacobians& jacs) const {
  const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
  if (v_->active()) {
    v_->backward(lhs * s_, child, jacs);
  }
}

// clang-format off
template <int ROW, int COL>
typename MatrixMultEvaluator<ROW, COL>::Ptr mmult(
    const typename Evaluable<typename MatrixMultEvaluator<ROW, COL>::InType>::ConstPtr& v,
    const typename MatrixMultEvaluator<ROW, COL>::MatType& s) {
  return MatrixMultEvaluator<ROW, COL>::MakeShared(v, s);
}
// clang-format on

}  // namespace vspace
}  // namespace steam