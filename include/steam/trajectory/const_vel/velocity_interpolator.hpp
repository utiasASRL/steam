#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_vel/variable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_vel {

class VelocityInterpolator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<VelocityInterpolator>;
  using ConstPtr = std::shared_ptr<const VelocityInterpolator>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Time time, const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2);
  VelocityInterpolator(const Time time, const Variable::ConstPtr& knot1,
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

}  // namespace const_vel
}  // namespace traj
}  // namespace steam
