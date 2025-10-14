#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_vel_se2/variable.hpp"

namespace steam {
namespace traj {
namespace const_vel_se2 {

class PriorFactor : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<PriorFactor>;
  using ConstPtr = std::shared_ptr<const PriorFactor>;

  using InPoseType = lgmath::se2::Transformation;
  using InVelType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2);
  PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

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
};

}  // namespace const_vel_se2
}  // namespace traj
}  // namespace steam
