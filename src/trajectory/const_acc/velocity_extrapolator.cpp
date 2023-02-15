#include "steam/trajectory/const_acc/velocity_extrapolator.hpp"
#include "steam/trajectory/const_acc/helper.hpp"

namespace steam {
namespace traj {
namespace const_acc {

auto VelocityExtrapolator::MakeShared(const Time& time,
                                      const Variable::ConstPtr& knot) -> Ptr {
  return std::make_shared<VelocityExtrapolator>(time, knot);
}

VelocityExtrapolator::VelocityExtrapolator(const Time& time,
                                           const Variable::ConstPtr& knot)
    : knot_(knot) {
  const double tau = (time - knot->time()).seconds();
  Phi_ = getTran(tau);
}

bool VelocityExtrapolator::active() const {
  return knot_->velocity()->active() || knot_->acceleration()->active();
}

void VelocityExtrapolator::getRelatedVarKeys(KeySet& keys) const {
  knot_->velocity()->getRelatedVarKeys(keys);
  knot_->acceleration()->getRelatedVarKeys(keys);
}

auto VelocityExtrapolator::value() const -> OutType {
  const Eigen::Matrix<double, 6, 1> xi_j1 =
      Phi_.block<6, 6>(6, 6) * knot_->velocity()->value() +
      Phi_.block<6, 6>(6, 12) * knot_->acceleration()->value();
  return OutType(xi_j1);  // approximation holds as long as xi_i1 is small.
}

auto VelocityExtrapolator::forward() const -> Node<OutType>::Ptr {
  const auto w = knot_->velocity()->forward();
  const auto dw = knot_->acceleration()->forward();
  const Eigen::Matrix<double, 6, 1> xi_j1 =
      Phi_.block<6, 6>(6, 6) * w->value() +
      Phi_.block<6, 6>(6, 12) * dw->value();
  OutType w_i = xi_j1;  // approximation holds as long as xi_i1 is small.
  const auto node = Node<OutType>::MakeShared(w_i);
  node->addChild(w);
  node->addChild(dw);
  return node;
}

void VelocityExtrapolator::backward(const Eigen::MatrixXd& lhs,
                                    const Node<OutType>::Ptr& node,
                                    Jacobians& jacs) const {
  if (!active()) return;
  if (knot_->velocity()->active()) {
    const auto w = std::static_pointer_cast<Node<InVelType>>(node->at(1));
    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(6, 6);
    knot_->velocity()->backward(new_lhs, w, jacs);
  }
  if (knot_->acceleration()->active()) {
    const auto dw = std::static_pointer_cast<Node<InAccType>>(node->at(1));
    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(6, 12);
    knot_->acceleration()->backward(new_lhs, dw, jacs);
  }
}

}  // namespace const_acc
}  // namespace traj
}  // namespace steam