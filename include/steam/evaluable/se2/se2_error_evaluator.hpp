#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se2 {

class SE2ErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<SE2ErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const SE2ErrorEvaluator>;

  using InType = lgmath::se2::Transformation;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_ab,
                        const InType& T_ab_meas);
  SE2ErrorEvaluator(const Evaluable<InType>::ConstPtr& T_ab,
                    const InType& T_ab_meas);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr T_ab_;
  const InType T_ab_meas_;
};

SE2ErrorEvaluator::Ptr se2_error(
    const Evaluable<SE2ErrorEvaluator::InType>::ConstPtr& T_ab,
    const SE2ErrorEvaluator::InType& T_ab_meas);

}  // namespace se2
}  // namespace steam