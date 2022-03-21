#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/traj_var.hpp"

namespace steam {
namespace traj {

/** \brief Gaussian-process prior evaluator */
class TrajPriorFactor : public Evaluable<Eigen::Matrix<double, 12, 1>> {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<TrajPriorFactor>;
  using ConstPtr = std::shared_ptr<const TrajPriorFactor>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 12, 1>;

  static Ptr MakeShared(const TrajVar::ConstPtr& knot1,
                        const TrajVar::ConstPtr& knot2);
  TrajPriorFactor(const TrajVar::ConstPtr& knot1,
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
};

}  // namespace traj
}  // namespace steam
