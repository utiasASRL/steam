#include "steam/trajectory/const_acc/acceleration_extrapolator.hpp"
#include "steam/trajectory/const_acc/helper.hpp"

namespace steam {
namespace traj {
namespace const_acc {

auto AccelerationExtrapolator::MakeShared(const Time& time,
                                          const Variable::ConstPtr& knot)
    -> Ptr {
  return std::make_shared<AccelerationExtrapolator>(time, knot);
}

AccelerationExtrapolator::AccelerationExtrapolator(
    const Time& time, const Variable::ConstPtr& knot)
    : knot_(knot) {
  const double tau = (time - knot->time()).seconds();
  Phi_ = getTran(tau);
}

bool AccelerationExtrapolator::active() const {
  return knot_->acceleration()->active();
}

void AccelerationExtrapolator::getRelatedVarKeys(KeySet& keys) const {
  knot_->acceleration()->getRelatedVarKeys(keys);
}

auto AccelerationExtrapolator::value() const -> OutType {
  const Eigen::Matrix<double, 6, 1> xi_k1 =
      Phi_.block<6, 6>(12, 12) * knot_->acceleration()->value();
  return OutType(xi_k1);  // approximation holds as long as xi_i1 is small.
}

auto AccelerationExtrapolator::forward() const -> Node<OutType>::Ptr {
  const auto dw = knot_->acceleration()->forward();
  const Eigen::Matrix<double, 6, 1> xi_k1 =
      Phi_.block<6, 6>(12, 12) * dw->value();
  OutType dw_i = xi_k1;  // approximation holds as long as xi_i1 is small.
  const auto node = Node<OutType>::MakeShared(dw_i);
  node->addChild(dw);
  return node;
}

void AccelerationExtrapolator::backward(const Eigen::MatrixXd& lhs,
                                        const Node<OutType>::Ptr& node,
                                        Jacobians& jacs) const {
  if (!active()) return;
  if (knot_->acceleration()->active()) {
    const auto dw = std::static_pointer_cast<Node<InAccType>>(node->at(0));
    Eigen::MatrixXd new_lhs = lhs * Phi_.block<6, 6>(12, 12);
    knot_->acceleration()->backward(new_lhs, dw, jacs);
  }
}

}  // namespace const_acc
}  // namespace traj
}  // namespace steam