#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se3 {

class InverseEvaluator : public Evaluable<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<InverseEvaluator>;
  using ConstPtr = std::shared_ptr<const InverseEvaluator>;

  using InType = lgmath::se3::Transformation;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform);
  InverseEvaluator(const Evaluable<InType>::ConstPtr& transform);

  bool active() const override;

  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr transform_;
};

InverseEvaluator::Ptr inverse(
    const Evaluable<InverseEvaluator::InType>::ConstPtr& transform);

}  // namespace se3
}  // namespace steam