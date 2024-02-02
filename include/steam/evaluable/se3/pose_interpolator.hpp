#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace se3 {

class PoseInterpolator : public Evaluable<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<PoseInterpolator>;
  using ConstPtr = std::shared_ptr<const PoseInterpolator>;
  using Time = steam::traj::Time;

  using InType = lgmath::se3::Transformation;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Time& time,
                        const Evaluable<InType>::ConstPtr& transform1,
                        const Time& time1,
                        const Evaluable<InType>::ConstPtr& transform2,
                        const Time& time2);
  PoseInterpolator(const Time& time,
                   const Evaluable<InType>::ConstPtr& transform1,
                   const Time& time1,
                   const Evaluable<InType>::ConstPtr& transform2,
                   const Time& time2);
  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr transform1_;
  const Evaluable<InType>::ConstPtr transform2_;
  double alpha_;
  std::vector<double> faulhaber_coeffs_;
};

}  // namespace se3
}  // namespace steam