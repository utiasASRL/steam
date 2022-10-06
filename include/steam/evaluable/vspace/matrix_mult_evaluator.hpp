//TODO: Need to generalize for arbitrary sized matrices
// Evaluator for left multiplying a static matrix with a vector space state variable 
// Currently used for turning a 2x1 unicycle velocity vector into an se3 velocity vector with a projection matrix
#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace vspace {

template <int DIM = Eigen::Dynamic>
class MatrixMultEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<MatrixMultEvaluator>;
  using ConstPtr = std::shared_ptr<const MatrixMultEvaluator>;

  using InType = Eigen::Matrix<double, DIM, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& v,
                        const Eigen::Matrix<double, 6, 2>& matrix);
  MatrixMultEvaluator(const typename Evaluable<InType>::ConstPtr& v,
                      const Eigen::Matrix<double, 6, 2>& matrix);

  bool active() const override;

  OutType value() const override;
  typename Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs,
                const typename Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const typename Evaluable<InType>::ConstPtr v_;
  const Eigen::Matrix<double, 6, 2> matrix_;
};

// clang-format off
template <int DIM>
typename MatrixMultEvaluator<DIM>::Ptr mmult(
    const typename Evaluable<typename MatrixMultEvaluator<DIM>::InType>::ConstPtr& v,
    const Eigen::Matrix<double, 6, 2>& matrix);
// clang-format on

}  // namespace vspace
}  // namespace steam

namespace steam {
namespace vspace {

template <int DIM>
auto MatrixMultEvaluator<DIM>::MakeShared(
    const typename Evaluable<InType>::ConstPtr& v, const Eigen::Matrix<double, 6, 2>& matrix) -> Ptr {
  return std::make_shared<MatrixMultEvaluator>(v, matrix);
}

template <int DIM>
MatrixMultEvaluator<DIM>::MatrixMultEvaluator(
    const typename Evaluable<InType>::ConstPtr& v, const Eigen::Matrix<double, 6, 2>& matrix)
    : v_(v), matrix_(matrix) {}

template <int DIM>
bool MatrixMultEvaluator<DIM>::active() const {
  return v_->active();
}

template <int DIM>
auto MatrixMultEvaluator<DIM>::value() const -> OutType {
  //return Eigen::Matrix<double, 6, 1>::Zero();
  return matrix_ * v_->value(); // Might need to actually remove the matrix_ mult here
}

template <int DIM>
auto MatrixMultEvaluator<DIM>::forward() const -> typename Node<OutType>::Ptr {
  const auto child = v_->forward();
  const auto value = matrix_ * child->value();
  //const auto value = Eigen::Matrix<double, 6, 1>::Zero();
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

template <int DIM>
void MatrixMultEvaluator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                        const typename Node<OutType>::Ptr& node,
                                        Jacobians& jacs) const {
  const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
  if (v_->active()) {
    v_->backward(lhs * matrix_, child, jacs); // Note I think for matrix mult we maybe flip the order here
  }
}

// clang-format off
template <int DIM>
typename MatrixMultEvaluator<DIM>::Ptr mmult(
    const typename Evaluable<typename MatrixMultEvaluator<DIM>::InType>::ConstPtr& v,
    const Eigen::Matrix<double, 6, 2>& matrix) {
  return MatrixMultEvaluator<DIM>::MakeShared(v, matrix);
}
// clang-format on

}  // namespace vspace
}  // namespace steam