#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_acc/variable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_acc {

class PoseInterpolator : public Evaluable<lgmath::se3::Transformation> {
 public:
  using Ptr = std::shared_ptr<PoseInterpolator>;
  using ConstPtr = std::shared_ptr<const PoseInterpolator>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using InAccType = Eigen::Matrix<double, 6, 1>;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2);
  PoseInterpolator(const Time& time, const Variable::ConstPtr& knot1,
                   const Variable::ConstPtr& knot2);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 protected:
  /** \brief First (earlier) knot */
  const Variable::ConstPtr knot1_;
  /** \brief Second (later) knot */
  const Variable::ConstPtr knot2_;
  /** \brief interpolation values **/
  Eigen::Matrix<double, 18, 18> omega_ = Eigen::Matrix<double, 18, 18>::Zero();
  Eigen::Matrix<double, 18, 18> lambda_ = Eigen::Matrix<double, 18, 18>::Zero();
};

}  // namespace const_acc
}  // namespace traj
}  // namespace steam