#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/traj_time.hpp"
#include "steam/trajectory/traj_var.hpp"

namespace steam {
namespace traj {

/** \brief Simple transform evaluator for a transformation state variable */
class TrajPoseExtrapolator : public Evaluable<lgmath::se3::Transformation> {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<TrajPoseExtrapolator>;
  using ConstPtr = std::shared_ptr<const TrajPoseExtrapolator>;

  using InType = Eigen::Matrix<double, 6, 1>;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Time& time,
                        const Evaluable<InType>::ConstPtr& velocity);
  TrajPoseExtrapolator(const Time& time,
                       const Evaluable<InType>::ConstPtr& velocity);

  bool active() const override;

  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief Duration */
  const Time time_;
  /** \brief Velocity state variable */
  const Evaluable<InType>::ConstPtr velocity_;
};

}  // namespace traj
}  // namespace steam
