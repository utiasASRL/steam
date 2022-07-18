#include "steam/trajectory/const_vel/pose_interpolator.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/trajectory/const_vel/evaluable/jinv_velocity_evaluator.hpp"

namespace steam {
namespace traj {
namespace const_vel {

PoseInterpolator::Ptr PoseInterpolator::MakeShared(
    const Time& time, const Variable::ConstPtr& knot1,
    const Variable::ConstPtr& knot2) {
  return std::make_shared<PoseInterpolator>(time, knot1, knot2);
}

PoseInterpolator::PoseInterpolator(const Time& time,
                                   const Variable::ConstPtr& knot1,
                                   const Variable::ConstPtr& knot2)
    : knot1_(knot1), knot2_(knot2) {
  // Calculate time constants
  double tau = (time - knot1->getTime()).seconds();
  double T = (knot2->getTime() - knot1->getTime()).seconds();
  double ratio = tau / T;
  double ratio2 = ratio * ratio;
  double ratio3 = ratio2 * ratio;

  // Calculate 'psi' interpolation values
  psi11_ = 3.0 * ratio2 - 2.0 * ratio3;
  psi12_ = tau * (ratio2 - ratio);
  psi21_ = 6.0 * (ratio - ratio2) / T;
  psi22_ = 3.0 * ratio2 - 2.0 * ratio;

  // Calculate 'lambda' interpolation values
  lambda11_ = 1.0 - psi11_;
  lambda12_ = tau - T * psi11_ - psi12_;
  lambda21_ = -psi21_;
  lambda22_ = 1.0 - T * psi21_ - psi22_;

  // construct computation graph
  const auto pose1 = knot1_->getPose();
  const auto vel1 = knot1_->getVelocity();
  const auto pose2 = knot2_->getPose();
  const auto vel2 = knot2_->getVelocity();

  using namespace steam::se3;
  using namespace steam::vspace;
  // Get relative matrix info
  const auto T_21 = compose_rinv(pose2, pose1);
  // Get se3 algebra of relative matrix
  const auto xi_21 = tran2vec(T_21);
  // Calculate interpolated relative se3 algebra
  const auto _t1 = smult<6>(vel1, lambda12_);
  const auto _t2 = smult<6>(xi_21, psi11_);
  const auto _t3 = smult<6>(jinv_velocity(xi_21, vel2), psi12_);
  const auto xi_i1 = add<6>(_t1, add<6>(_t2, _t3));
  // calculate interpolated relative transformation matrix
  const auto T_i1 = vec2tran(xi_i1);
  // compose to get global transform
  T_i0_ = compose(T_i1, pose1);
}

bool PoseInterpolator::active() const {
  return knot1_->getPose()->active() || knot1_->getVelocity()->active() ||
         knot2_->getPose()->active() || knot2_->getVelocity()->active();
}

void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
  knot1_->getPose()->getRelatedVarKeys(keys);
  knot1_->getVelocity()->getRelatedVarKeys(keys);
  knot2_->getPose()->getRelatedVarKeys(keys);
  knot2_->getVelocity()->getRelatedVarKeys(keys);
}

auto PoseInterpolator::value() const -> OutType { return T_i0_->value(); }

auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
  return T_i0_->forward();
}

void PoseInterpolator::backward(const Eigen::MatrixXd& lhs,
                                const Node<OutType>::Ptr& node,
                                Jacobians& jacs) const {
  return T_i0_->backward(lhs, node, jacs);
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam