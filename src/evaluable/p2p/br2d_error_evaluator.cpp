#include "steam/evaluable/p2p/br2d_error_evaluator.hpp"

namespace steam {
namespace p2p {

auto BRError2DEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& point,
                                    const Eigen::Vector2d& br_meas) -> Ptr {
  return std::make_shared<BRError2DEvaluator>(point, br_meas);
}

BRError2DEvaluator::BRError2DEvaluator(const Evaluable<InType>::ConstPtr& point,
                                       const Eigen::Vector2d& br_meas)
    : point_(point), br_meas_(br_meas) {}

bool BRError2DEvaluator::active() const { return point_->active(); }

void BRError2DEvaluator::getRelatedVarKeys(KeySet& keys) const {
  return point_->getRelatedVarKeys(keys);
}

auto BRError2DEvaluator::value() const -> OutType {
  double x = point_->value()[0];
  double y = point_->value()[1];
  const auto br_val = Eigen::Vector2d(atan2(y, x), sqrt(pow(x, 2) + pow(y, 2)));
  return br_val - br_meas_;
}
auto BRError2DEvaluator::forward() const -> Node<OutType>::Ptr {
  // Call forward of child point node
  const auto child = point_->forward();
  // get value
  double x = child->value()[0];
  double y = child->value()[1];
  OutType value =
      Eigen::Vector2d(atan2(y, x), sqrt(pow(x, 2) + pow(y, 2))) - br_meas_;
  // create node
  const auto node = Node<OutType>::MakeShared(value);
  node->addChild(child);
  return node;
}

void BRError2DEvaluator::backward(const Eigen::MatrixXd& lhs,
                                  const Node<OutType>::Ptr& node,
                                  Jacobians& jacs) const {
  // get child node pointer
  const auto child = std::static_pointer_cast<Node<InType>>(node->at(0));
  // build jacobian
  if (point_->active()) {
    // Retrieve range
    double x = child->value()[0];
    double y = child->value()[1];
    double r = sqrt(pow(x, 2) + pow(y, 2));
    // compute jacobian
    auto Jac = Eigen::Matrix<double, 2, 4>::Zero().eval();
    if (r > 1e-8) {
      Jac(0, 0) = -y / pow(r, 2);
      Jac(0, 1) = x / pow(r, 2);
      Jac(1, 0) = x / r;
      Jac(1, 1) = y / r;
    }
    point_->backward(lhs * Jac, child, jacs);
  }
}

BRError2DEvaluator::Ptr br2dError(
    const Evaluable<BRError2DEvaluator::InType>::ConstPtr& point,
    const Eigen::Vector2d& br_meas) {
  return BRError2DEvaluator::MakeShared(point, br_meas);
}

}  // namespace p2p
}  // namespace steam