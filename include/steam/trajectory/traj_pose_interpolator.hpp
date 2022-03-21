#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/traj_time.hpp"
#include "steam/trajectory/traj_var.hpp"

namespace steam {
namespace traj {

class TrajPoseInterpolator : public Evaluable<lgmath::se3::Transformation> {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<TrajPoseInterpolator>;
  using ConstPtr = std::shared_ptr<const TrajPoseInterpolator>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using OutType = lgmath::se3::Transformation;

  static Ptr MakeShared(const Time& time, const TrajVar::ConstPtr& knot1,
                        const TrajVar::ConstPtr& knot2);
  TrajPoseInterpolator(const Time& time, const TrajVar::ConstPtr& knot1,
                       const TrajVar::ConstPtr& knot2);

  bool active() const override;

  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief First (earlier) knot */
  const TrajVar::ConstPtr knot1_;
  /** \brief Second (later) knot */
  const TrajVar::ConstPtr knot2_;

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

}  // namespace traj
}  // namespace steam
