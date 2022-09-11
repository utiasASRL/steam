#include "steam/trajectory/const_vel/prior_factor.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/trajectory/const_vel/evaluable/jinv_velocity_evaluator.hpp"
#include "steam/trajectory/const_vel/helper.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto PriorFactor::MakeShared(const Variable::ConstPtr& knot1,
                             const Variable::ConstPtr& knot2) -> Ptr {
  return std::make_shared<PriorFactor>(knot1, knot2);
}

PriorFactor::PriorFactor(const Variable::ConstPtr& knot1,
                         const Variable::ConstPtr& knot2)
    : knot1_(knot1), knot2_(knot2) {
  // constants
  const double dt = (knot2->time() - knot1->time()).seconds();

  // construct computation graph
  const auto T1 = knot1_->pose();
  const auto w1 = knot1_->velocity();
  const auto T2 = knot2_->pose();
  const auto w2 = knot2_->velocity();

  using namespace steam::se3;
  using namespace steam::vspace;

  // get relative matrix info
  const auto T_21 = compose_rinv(T2, T1);
  // get se3 algebra of relative matrix
  const auto xi_21 = tran2vec(T_21);

  // pose error
  const auto t1_ = xi_21;
  const auto t2_ = smult<6>(w1, -dt);
  ep_ = add<6>(t1_, t2_);

  // velocity error
  const auto w1_ = jinv_velocity(xi_21, w2);
  const auto w2_ = neg<6>(w1);
  ev_ = add<6>(w1_, w2_);
}

bool PriorFactor::active() const {
  return knot1_->pose()->active() || knot1_->velocity()->active() ||
         knot2_->pose()->active() || knot2_->velocity()->active();
}

void PriorFactor::getRelatedVarKeys(KeySet& keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
}

auto PriorFactor::value() const -> OutType {
  //
  OutType error = OutType::Zero();
  error.block<6, 1>(0, 0) = ep_->value();
  error.block<6, 1>(6, 0) = ev_->value();
  return error;
}

auto PriorFactor::forward() const -> Node<OutType>::Ptr {
  //
  const auto ep = ep_->forward();
  const auto ev = ev_->forward();

  //
  OutType error = OutType::Zero();
  error.block<6, 1>(0, 0) = ep_->value();
  error.block<6, 1>(6, 0) = ev_->value();

  //
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(ep);
  node->addChild(ev);

  return node;
}

void PriorFactor::backward(const Eigen::MatrixXd& lhs,
                           const Node<OutType>::Ptr& node,
                           Jacobians& jacs) const {
  using OutT = Eigen::Matrix<double, 6, 1>;
  if (ep_->active()) {
    const auto ep = std::static_pointer_cast<Node<OutT>>(node->at(0));
    ep_->backward(lhs.leftCols(6), ep, jacs);
  }

  if (ev_->active()) {
    const auto ev = std::static_pointer_cast<Node<OutT>>(node->at(1));
    ev_->backward(lhs.rightCols(6), ev, jacs);
  }
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam