#include "steam/trajectory/const_acc/prior_factor.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/trajectory/const_acc/evaluable/compose_curlyhat_evaluator.hpp"
#include "steam/trajectory/const_vel/evaluable/jinv_velocity_evaluator.hpp"

namespace steam {
namespace traj {
namespace const_acc {

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
  const auto dw1 = knot1_->acceleration();
  const auto T2 = knot2_->pose();
  const auto w2 = knot2_->velocity();
  const auto dw2 = knot2_->acceleration();

  using namespace steam::se3;
  using namespace steam::vspace;

  // get relative matrix info
  const auto T_21 = compose_rinv(T2, T1);
  // get se3 algebra of relative matrix
  const auto xi_21 = tran2vec(T_21);

  // pose error
  const auto t1_ = xi_21;
  const auto t2_ = smult<6>(w1, -dt);
  const auto t3_ = smult<6>(dw1, -0.5 * dt * dt);
  ep_ = add<6>(t1_, add<6>(t2_, t3_));

  // velocity error
  const auto w1_ = const_vel::jinv_velocity(xi_21, w2);
  const auto w2_ = neg<6>(w1);
  const auto w3_ = smult<6>(dw1, -dt);
  ev_ = add<6>(w1_, add<6>(w2_, w3_));

  // clang-format off
  // acceleration error
  const auto dw1_ = smult<6>(compose_curlyhat(const_vel::jinv_velocity(xi_21, dw2), dw2), -0.5);
  const auto dw2_ = const_vel::jinv_velocity(xi_21, dw2);
  const auto dw3_ = neg<6>(dw1);
  ea_ = add<6>(dw1_, add<6>(dw2_, dw3_));
  // clang-format on
}

bool PriorFactor::active() const {
  return knot1_->pose()->active() || knot1_->velocity()->active() ||
         knot1_->acceleration()->active() || knot2_->pose()->active() ||
         knot2_->velocity()->active() || knot2_->acceleration()->active();
}

void PriorFactor::getRelatedVarKeys(KeySet& keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot1_->acceleration()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
  knot2_->acceleration()->getRelatedVarKeys(keys);
}

auto PriorFactor::value() const -> OutType {
  //
  OutType error = OutType::Zero();
  error.block<6, 1>(0, 0) = ep_->value();
  error.block<6, 1>(6, 0) = ev_->value();
  error.block<6, 1>(12, 0) = ea_->value();
  return error;
}

auto PriorFactor::forward() const -> Node<OutType>::Ptr {
  //
  const auto ep = ep_->forward();
  const auto ev = ev_->forward();
  const auto ea = ea_->forward();

  //
  OutType error = OutType::Zero();
  error.block<6, 1>(0, 0) = ep_->value();
  error.block<6, 1>(6, 0) = ev_->value();
  error.block<6, 1>(12, 0) = ea_->value();

  //
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(ep);
  node->addChild(ev);
  node->addChild(ea);

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
    ev_->backward(lhs.middleCols(6, 6), ev, jacs);
  }

  if (ea_->active()) {
    const auto ea = std::static_pointer_cast<Node<OutT>>(node->at(2));
    ea_->backward(lhs.rightCols(6), ea, jacs);
  }
}

}  // namespace const_acc
}  // namespace traj
}  // namespace steam