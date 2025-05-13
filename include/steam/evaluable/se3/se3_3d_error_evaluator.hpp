#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se3 {

class SE33DErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<SE33DErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const SE33DErrorEvaluator>;

  using InType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_ab);
  SE33DErrorEvaluator(const Evaluable<InType>::ConstPtr& T_ab);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr T_ab_;
  const InType T_ab_meas_;
  Eigen::Matrix<double, 3, 6> D_;
};

SE33DErrorEvaluator::Ptr se3_3d_error(
    const Evaluable<SE33DErrorEvaluator::InType>::ConstPtr& T_ab);

}  // namespace se3
}  // namespace steam