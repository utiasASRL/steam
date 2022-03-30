#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_vel/variable.hpp"

namespace steam {
namespace traj {
namespace const_vel {

/** \brief Gaussian-process prior evaluator */
class PriorFactor : public Evaluable<Eigen::Matrix<double, 12, 1>> {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<PriorFactor>;
  using ConstPtr = std::shared_ptr<const PriorFactor>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 12, 1>;

  static Ptr MakeShared(const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2);
  PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

  bool active() const override;

  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief First (earlier) knot */
  const Variable::ConstPtr knot1_;
  /** \brief Second (later) knot */
  const Variable::ConstPtr knot2_;
};

}  // namespace const_vel
}  // namespace traj
}  // namespace steam
