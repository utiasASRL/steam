#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace traj {
namespace const_acc {

class ComposeCurlyhatEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<ComposeCurlyhatEvaluator>;
  using ConstPtr = std::shared_ptr<const ComposeCurlyhatEvaluator>;

  using InType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& x,
                        const Evaluable<InType>::ConstPtr& y);
  ComposeCurlyhatEvaluator(const Evaluable<InType>::ConstPtr& x,
                           const Evaluable<InType>::ConstPtr& y);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr x_;
  const Evaluable<InType>::ConstPtr y_;
};

ComposeCurlyhatEvaluator::Ptr compose_curlyhat(
    const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& x,
    const Evaluable<ComposeCurlyhatEvaluator::InType>::ConstPtr& y);

}  // namespace const_acc
}  // namespace traj
}  // namespace steam