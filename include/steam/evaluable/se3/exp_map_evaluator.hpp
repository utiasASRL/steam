#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se3 {

class ExpMapEvaluator : public Evaluable<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<ExpMapEvaluator>;
  using ConstPtr = std::shared_ptr<const ExpMapEvaluator>;

  using InType = Eigen::Matrix<double, 6, 1>;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& xi);
  ExpMapEvaluator(const Evaluable<InType>::ConstPtr& xi);

  bool active() const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr xi_;
};

ExpMapEvaluator::Ptr vec2tran(
    const Evaluable<ExpMapEvaluator::InType>::ConstPtr& xi);

}  // namespace se3
}  // namespace steam