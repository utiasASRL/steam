#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se2 {

class ComposeInverseEvaluator : public Evaluable<lgmath::se2::Transformation> {
 public:
  using Ptr = std::shared_ptr<ComposeInverseEvaluator>;
  using ConstPtr = std::shared_ptr<const ComposeInverseEvaluator>;

  using InType = lgmath::se2::Transformation;
  using OutType = lgmath::se2::Transformation;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform1,
                        const Evaluable<InType>::ConstPtr& transform2);
  ComposeInverseEvaluator(const Evaluable<InType>::ConstPtr& transform1,
                          const Evaluable<InType>::ConstPtr& transform2);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr transform1_;
  const Evaluable<InType>::ConstPtr transform2_;
};

ComposeInverseEvaluator::Ptr compose_rinv(
    const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr& transform1,
    const Evaluable<ComposeInverseEvaluator::InType>::ConstPtr& transform2);

}  // namespace se2
}  // namespace steam