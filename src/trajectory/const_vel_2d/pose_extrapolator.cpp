#include "steam/trajectory/const_vel_2d/pose_extrapolator.hpp"
#include "steam/trajectory/const_vel_2d/helper.hpp"

namespace steam {
namespace traj {
namespace const_vel_2d {

auto PoseExtrapolator::MakeShared(const Time time,
                                  const Variable::ConstPtr& knot) -> Ptr {
  return std::make_shared<PoseExtrapolator>(time, knot);
}

PoseExtrapolator::PoseExtrapolator(const Time time,
                                   const Variable::ConstPtr& knot)
    : knot_(knot) {
  const double tau = (time - knot->time()).seconds();
  Phi_ = getTran(tau);
}

bool PoseExtrapolator::active() const {
  return knot_->pose()->active() || knot_->velocity()->active();
}

void PoseExtrapolator::getRelatedVarKeys(KeySet& keys) const {
  knot_->pose()->getRelatedVarKeys(keys);
  knot_->velocity()->getRelatedVarKeys(keys);
}

auto PoseExtrapolator::value() const -> OutType {
  const lgmath::se2::Transformation T_i1(Eigen::Matrix<double, 3, 1>(
      Phi_.block<3, 3>(0, 3) * knot_->velocity()->value()));
  return OutType(T_i1 * knot_->pose()->value());
}

auto PoseExtrapolator::forward() const -> Node<OutType>::Ptr {
  const auto T = knot_->pose()->forward();
  const auto w = knot_->velocity()->forward();
  const lgmath::se2::Transformation T_i1(
      Eigen::Matrix<double, 3, 1>(Phi_.block<3, 3>(0, 3) * w->value()));
  OutType T_i0 = T_i1 * T->value();
  const auto node = Node<OutType>::MakeShared(T_i0);
  node->addChild(T);
  node->addChild(w);
  return node;
}

void PoseExtrapolator::backward(const Eigen::MatrixXd& lhs,
                                const Node<OutType>::Ptr& node,
                                Jacobians& jacs) const {
  if (!active()) return;
  const auto w = knot_->velocity()->value();
  const Eigen::Matrix<double, 3, 1> xi_i1 = Phi_.block<3, 3>(0, 3) * w;
  const Eigen::Matrix<double, 3, 3> J_i1 = lgmath::se2::vec2jac(xi_i1);
  const lgmath::se2::Transformation T_i1(xi_i1);
  if (knot_->pose()->active()) {
    const auto T_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
    Eigen::MatrixXd new_lhs = lhs * T_i1.adjoint();
    knot_->pose()->backward(new_lhs, T_, jacs);
  }
  if (knot_->velocity()->active()) {
    const auto w_ = std::static_pointer_cast<Node<InVelType>>(node->at(1));
    Eigen::MatrixXd new_lhs = lhs * J_i1 * Phi_.block<3, 3>(0, 3);
    knot_->velocity()->backward(new_lhs, w_, jacs);
  }
}

}  // namespace const_vel_2d
}  // namespace traj
}  // namespace steam