#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_vel/variable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_vel {

/** \brief Simple transform evaluator for a transformation state variable */
class PoseExtrapolator : public Evaluable<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<PoseExtrapolator>;
  using ConstPtr = std::shared_ptr<const PoseExtrapolator>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot);
  PoseExtrapolator(const Time& time, const Variable::ConstPtr& knot);
  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;
  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief Knot to extrapolate from */
  const Variable::ConstPtr knot_;
  /** \brief Transition matrix */
  Eigen::Matrix<double, 12, 12> Phi_;
};

}  // namespace const_vel
}  // namespace traj
}  // namespace steam
