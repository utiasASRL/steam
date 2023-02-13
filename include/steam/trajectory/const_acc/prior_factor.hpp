#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/const_acc/variable.hpp"

namespace steam {
namespace traj {
namespace const_acc {

class PriorFactor : public Evaluable<Eigen::Matrix<double, 18, 1>> {
 public:
  using Ptr = std::shared_ptr<PriorFactor>;
  using ConstPtr = std::shared_ptr<const PriorFactor>;

  using InPoseType = lgmath::se3::Transformation;
  using InVelType = Eigen::Matrix<double, 6, 1>;
  using InAccType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 18, 1>;

  static Ptr MakeShared(const Variable::ConstPtr& knot1,
                        const Variable::ConstPtr& knot2);
  PriorFactor(const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2);

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
  /** \brief Transition matrix */
  Eigen::Matrix<double, 18, 18> Phi_ = Eigen::Matrix<double, 18, 18>::Identity();
  Eigen::Matrix<double, 18, 18> getJacKnot1_() const;
  Eigen::Matrix<double, 18, 18> getJacKnot2_() const;
};

}  // namespace const_acc
}  // namespace traj
}  // namespace steam
