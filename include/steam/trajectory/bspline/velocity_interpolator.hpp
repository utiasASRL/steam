#pragma once

#include <Eigen/Core>

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/bspline/variable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace bspline {

class VelocityInterpolator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<VelocityInterpolator>;
  using ConstPtr = std::shared_ptr<const VelocityInterpolator>;

  using CType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Time& time, const Variable::ConstPtr& k1,
                        const Variable::ConstPtr& k2,
                        const Variable::ConstPtr& k3,
                        const Variable::ConstPtr& k4);
  VelocityInterpolator(const Time& time, const Variable::ConstPtr& k1,
                       const Variable::ConstPtr& k2,
                       const Variable::ConstPtr& k3,
                       const Variable::ConstPtr& k4);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Variable::ConstPtr k1_;
  const Variable::ConstPtr k2_;
  const Variable::ConstPtr k3_;
  const Variable::ConstPtr k4_;

  Eigen::Matrix<double, 4, 1> w_;
};

}  // namespace bspline
}  // namespace traj
}  // namespace steam
