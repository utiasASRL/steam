#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se3 {

class SE3ErrorGlobalPerturbEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<SE3ErrorGlobalPerturbEvaluator>;
  using ConstPtr = std::shared_ptr<const SE3ErrorGlobalPerturbEvaluator>;

  using InType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& T_ab,
                        const InType& T_ab_meas);
  SE3ErrorGlobalPerturbEvaluator(const Evaluable<InType>::ConstPtr& T_ab,
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

SE3ErrorGlobalPerturbEvaluator::Ptr se3_global_perturb_error(
    const Evaluable<SE3ErrorGlobalPerturbEvaluator::InType>::ConstPtr& T_ab,
    const SE3ErrorGlobalPerturbEvaluator::InType& T_ab_meas);

}  // namespace se3
}  // namespace steam