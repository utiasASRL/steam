#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se3 {

class SE3ErrorEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<SE3ErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const SE3ErrorEvaluator>;

  using InType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_ab,
                        const InType& T_ab_meas);
  SE3ErrorEvaluator(const Evaluable<InType>::ConstPtr& T_ab,
                    const InType& T_ab_meas);

  bool active() const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr T_ab_;
  const InType T_ab_meas_;
};

SE3ErrorEvaluator::Ptr se3_error(
    const Evaluable<SE3ErrorEvaluator::InType>::ConstPtr& T_ab,
    const SE3ErrorEvaluator::InType& T_ab_meas);

}  // namespace se3
}  // namespace steam