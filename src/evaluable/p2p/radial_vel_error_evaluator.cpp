#include "steam/evaluable/p2p/radial_vel_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto RadialVelErrorEvaluator::MakeShared(
    const Evaluable<InType>::ConstPtr &w_iv_inv, const Eigen::Vector3d &pv,
    const double &r) -> Ptr {
  return std::make_shared<RadialVelErrorEvaluator>(w_iv_inv, pv, r);
}

RadialVelErrorEvaluator::RadialVelErrorEvaluator(
    const Evaluable<InType>::ConstPtr &w_iv_inv, const Eigen::Vector3d &pv,
    const double &r)
    : w_iv_inv_(w_iv_inv), pv_(pv), r_{r} {
  D_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
}

bool RadialVelErrorEvaluator::active() const { return w_iv_inv_->active(); }

auto RadialVelErrorEvaluator::value() const -> OutType {
  const Eigen::Matrix<double, 1, 1> numerator =
      (pv_.transpose() * D_ * lgmath::se3::point2fs(pv_, 1.0) *
       w_iv_inv_->value());
  const double denominator = std::sqrt(double(pv_.transpose() * pv_));
  return r_ - numerator / denominator;
}

auto RadialVelErrorEvaluator::forward() const -> Node<OutType>::Ptr {
  const auto child = w_iv_inv_->forward();
  //
  const Eigen::Matrix<double, 1, 1> numerator =
      pv_.transpose() * D_ * lgmath::se3::point2fs(pv_, 1.0) * child->value();
  const double denominator = std::sqrt(double(pv_.transpose() * pv_));
  const Eigen::Matrix<double, 1, 1> error = r_ - numerator / denominator;
  //
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(child);
  return node;
}

void RadialVelErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                       const Node<OutType>::Ptr &node,
                                       Jacobians &jacs) const {
  if (w_iv_inv_->active()) {
    const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
    const auto jac_unnorm =
        pv_.transpose() * D_ * lgmath::se3::point2fs(pv_, 1.0);
    const double pv_norm = std::sqrt(double(pv_.transpose() * pv_));
    const auto jac = jac_unnorm / pv_norm;
    w_iv_inv_->backward(-lhs * jac, child, jacs);
  }
}

RadialVelErrorEvaluator::Ptr radialVelError(
    const Evaluable<RadialVelErrorEvaluator::InType>::ConstPtr &w_iv_inv,
    const Eigen::Vector3d &pv, const double &r) {
  return RadialVelErrorEvaluator::MakeShared(w_iv_inv, pv, r);
}

}  // namespace p2p
}  // namespace steam