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
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<VelocityInterpolator>;
  using ConstPtr = std::shared_ptr<const VelocityInterpolator>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2);
  VelocityInterpolator(const Time& time, const Variable::ConstPtr& knot1,
                       const Variable::ConstPtr& knot2);

  bool active() const override;

  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief First (earlier) knot */
  const Variable::ConstPtr knot1_;
  /** \brief Second (later) knot */
  const Variable::ConstPtr knot2_;

  /** \brief Interpolation coefficients */
  double psi11_;
  double psi12_;
  double psi21_;
  double psi22_;
  double lambda11_;
  double lambda12_;
  double lambda21_;
  double lambda22_;
};

}  // namespace const_vel
}  // namespace traj
}  // namespace steam
