#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_vel_2d/variable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_vel_2d {

class PoseInterpolator : public Evaluable<lgmath::se2::Transformation> {
 public:
  using Ptr = std::shared_ptr<PoseInterpolator>;
  using ConstPtr = std::shared_ptr<const PoseInterpolator>;

  using InPoseType = lgmath::se2::Transformation;
  using InVelType = Eigen::Matrix<double, 3, 1>;
  using OutType = lgmath::se2::Transformation;

  static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2);
  PoseInterpolator(const Time time, const Variable::ConstPtr& knot1,
                   const Variable::ConstPtr& knot2);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief First (earlier) knot */
  const Variable::ConstPtr knot1_;
  /** \brief Second (later) knot */
  const Variable::ConstPtr knot2_;
  /** \brief interpolation values **/
  double psi11_, psi12_, psi21_, psi22_, lambda11_, lambda12_, lambda21_,
      lambda22_;
};

}  // namespace const_vel_2d
}  // namespace traj
}  // namespace steam