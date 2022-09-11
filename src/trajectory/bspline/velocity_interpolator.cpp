#include "steam/trajectory/bspline/velocity_interpolator.hpp"

#include "steam/evaluable/vspace/evaluables.hpp"

namespace steam {
namespace traj {
namespace bspline {

VelocityInterpolator::Ptr VelocityInterpolator::MakeShared(
    const Time& time, const Variable::ConstPtr& k1,
    const Variable::ConstPtr& k2, const Variable::ConstPtr& k3,
    const Variable::ConstPtr& k4) {
  return std::make_shared<VelocityInterpolator>(time, k1, k2, k3, k4);
}

VelocityInterpolator::VelocityInterpolator(const Time& time,
                                           const Variable::ConstPtr& k1,
                                           const Variable::ConstPtr& k2,
                                           const Variable::ConstPtr& k3,
                                           const Variable::ConstPtr& k4)
    : k1_(k1), k2_(k2), k3_(k3), k4_(k4) {
  // Calculate time constants
  double tau = (time - k2_->getTime()).seconds();
  double T = (k3_->getTime() - k2_->getTime()).seconds();
  double ratio = tau / T;
  double ratio2 = ratio * ratio;
  double ratio3 = ratio2 * ratio;

  // clang-format off
  Eigen::Matrix4d B;
  B << 1.,  4.,  1., 0.,
      -3.,  0.,  3., 0.,
       3., -6.,  3., 0.,
      -1.,  3., -3., 1.;
  B /= 6.0;
  // clang-format on
  Eigen::Vector4d u;
  u << 1., ratio, ratio2, ratio3;

  w_ = (u.transpose() * B).transpose();
}

bool VelocityInterpolator::active() const {
  return k1_->getC()->active() || k2_->getC()->active() ||
         k3_->getC()->active() || k4_->getC()->active();
}

void VelocityInterpolator::getRelatedVarKeys(KeySet& keys) const {
  k1_->getC()->getRelatedVarKeys(keys);
  k2_->getC()->getRelatedVarKeys(keys);
  k3_->getC()->getRelatedVarKeys(keys);
  k4_->getC()->getRelatedVarKeys(keys);
}

auto VelocityInterpolator::value() const -> OutType {
  return w_(0) * k1_->getC()->value() + w_(1) * k2_->getC()->value() +
         w_(2) * k3_->getC()->value() + w_(3) * k4_->getC()->value();
}

auto VelocityInterpolator::forward() const -> Node<OutType>::Ptr {
  const auto k1 = k1_->getC()->forward();
  const auto k2 = k2_->getC()->forward();
  const auto k3 = k3_->getC()->forward();
  const auto k4 = k4_->getC()->forward();

  const auto value = w_(0) * k1->value() + w_(1) * k2->value() +
                     w_(2) * k3->value() + w_(3) * k4->value();

  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(k1);
  node->addChild(k2);
  node->addChild(k3);
  node->addChild(k4);

  return node;
}

void VelocityInterpolator::backward(const Eigen::MatrixXd& lhs,
                                    const Node<OutType>::Ptr& node,
                                    Jacobians& jacs) const {
  if (k1_->getC()->active()) {
    const auto child = std::static_pointer_cast<Node<CType>>(node->at(0));
    k1_->getC()->backward(lhs * w_(0), child, jacs);
  }
  if (k2_->getC()->active()) {
    const auto child = std::static_pointer_cast<Node<CType>>(node->at(1));
    k2_->getC()->backward(lhs * w_(1), child, jacs);
  }
  if (k3_->getC()->active()) {
    const auto child = std::static_pointer_cast<Node<CType>>(node->at(2));
    k3_->getC()->backward(lhs * w_(2), child, jacs);
  }
  if (k4_->getC()->active()) {
    const auto child = std::static_pointer_cast<Node<CType>>(node->at(3));
    k4_->getC()->backward(lhs * w_(3), child, jacs);
  }
}

}  // namespace bspline
}  // namespace traj
}  // namespace steam