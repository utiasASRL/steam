#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_acc/variable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_acc {

class PoseExtrapolator : public Evaluable<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<PoseExtrapolator>;
  using ConstPtr = std::shared_ptr<const PoseExtrapolator>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using InAccType = Eigen::Matrix<double, 6, 1>;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot);
  PoseExtrapolator(const Time& time, const Variable::ConstPtr& knot);
  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;
  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 protected:
  /** \brief Knot to extrapolate from */
  const Variable::ConstPtr knot_;
  /** \brief Transition matrix */
  Eigen::Matrix<double, 18, 18> Phi_;
};

}  // namespace const_acc
}  // namespace traj
}  // namespace steam
