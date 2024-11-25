#include "steam/trajectory/const_acc/pose_extrapolator.hpp"
#include "steam/trajectory/const_acc/helper.hpp"

namespace steam {
namespace traj {
namespace const_acc {

PoseExtrapolator::Ptr PoseExtrapolator::MakeShared(
    const Time time, const Variable::ConstPtr& knot) {
  return std::make_shared<PoseExtrapolator>(time, knot);
}

PoseExtrapolator::PoseExtrapolator(const Time time,
                                   const Variable::ConstPtr& knot)
    : knot_(knot) {
  const double tau = (time - knot->time()).seconds();
  Phi_ = getTran(tau);
}

bool PoseExtrapolator::active() const {
  return knot_->pose()->active() || knot_->velocity()->active() ||
         knot_->acceleration()->active();
}

void PoseExtrapolator::getRelatedVarKeys(KeySet& keys) const {
  knot_->pose()->getRelatedVarKeys(keys);
  knot_->velocity()->getRelatedVarKeys(keys);
  knot_->acceleration()->getRelatedVarKeys(keys);
}

auto PoseExtrapolator::value() const -> OutType {
  const lgmath::se3::Transformation T_i1(Eigen::Matrix<double, 6, 1>(
      Phi_.block<6, 6>(0, 6) * knot_->velocity()->value() +
      Phi_.block<6, 6>(0, 12) * knot_->acceleration()->value()));
  return OutType(T_i1 * knot_->pose()->value());
}

auto PoseExtrapolator::forward() const -> Node<OutType>::Ptr {
  const auto T = knot_->pose()->forward();
  const auto w = knot_->velocity()->forward();
  const auto dw = knot_->acceleration()->forward();
  const lgmath::se3::Transformation T_i1(
      Eigen::Matrix<double, 6, 1>(Phi_.block<6, 6>(0, 6) * w->value() +
                                  Phi_.block<6, 6>(0, 12) * dw->value()));
  OutType T_i0 = T_i1 * T->value();
  const auto node = Node<OutType>::MakeShared(T_i0);
  node->addChild(T);
  node->addChild(w);
  node->addChild(dw);
  return node;
}

void PoseExtrapolator::backward(const Eigen::MatrixXd& lhs,
                                const Node<OutType>::Ptr& node,
                                Jacobians& jacs) const {
  if (!active()) return;
  const auto w = knot_->velocity()->value();
  const auto dw = knot_->acceleration()->value();
  const Eigen::Matrix<double, 6, 1> xi_i1 =
      Phi_.block<6, 6>(0, 6) * w + Phi_.block<6, 6>(0, 12) * dw;
  const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);
  const lgmath::se3::Transformation T_i1(xi_i1);
  if (knot_->pose()->active()) {
    const auto T_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
    Eigen::MatrixXd new_lhs = lhs * T_i1.adjoint();
    knot_->pose()->backward(new_lhs, T_, jacs);
  }
  if (knot_->velocity()->active()) {
    const auto w = std::static_pointer_cast<Node<InVelType>>(node->at(1));
    Eigen::MatrixXd new_lhs = lhs * J_i1 * Phi_.block<6, 6>(0, 6);
    knot_->velocity()->backward(new_lhs, w, jacs);
  }
  if (knot_->acceleration()->active()) {
    const auto dw = std::static_pointer_cast<Node<InAccType>>(node->at(2));
    Eigen::MatrixXd new_lhs = lhs * J_i1 * Phi_.block<6, 6>(0, 12);
    knot_->acceleration()->backward(new_lhs, dw, jacs);
  }
}

}  // namespace const_acc
}  // namespace traj
}  // namespace steam