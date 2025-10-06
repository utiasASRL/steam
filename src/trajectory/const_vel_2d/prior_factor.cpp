#include "steam/trajectory/const_vel_2d/prior_factor.hpp"
#include <iostream>

#include "steam/evaluable/se2/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/trajectory/const_vel_2d/evaluable/jinv_velocity_evaluator.hpp"
#include "steam/trajectory/const_vel_2d/helper.hpp"

namespace steam {
namespace traj {
namespace const_vel_2d {

auto PriorFactor::MakeShared(const Variable::ConstPtr& knot1,
                             const Variable::ConstPtr& knot2) -> Ptr {
  return std::make_shared<PriorFactor>(knot1, knot2);
}

PriorFactor::PriorFactor(const Variable::ConstPtr& knot1,
                         const Variable::ConstPtr& knot2)
    : knot1_(knot1), knot2_(knot2) {}

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
  OutType error = OutType::Zero();

  const auto T1 = knot1_->pose()->value();
  const auto w1 = knot1_->velocity()->value();
  const auto T2 = knot2_->pose()->value();
  const auto w2 = knot2_->velocity()->value();

  const double dt = (knot2_->time() - knot1_->time()).seconds();
  const auto xi_21 = (T2 / T1).vec();
  error.block<3, 1>(0, 0) = xi_21 - dt * w1;
  error.block<3, 1>(3, 0) = lgmath::se2::vec2jacinv(xi_21) * w2 - w1;
  return error;
}

auto PriorFactor::forward() const -> Node<OutType>::Ptr {
  const auto T1 = knot1_->pose()->forward();
  const auto w1 = knot1_->velocity()->forward();
  const auto T2 = knot2_->pose()->forward();
  const auto w2 = knot2_->velocity()->forward();

  const double dt = (knot2_->time() - knot1_->time()).seconds();
  const auto xi_21 = (T2->value() / T1->value()).vec();
  OutType error = OutType::Zero();
  error.block<3, 1>(0, 0) = xi_21 - dt * w1->value();
  error.block<3, 1>(3, 0) =
      lgmath::se2::vec2jacinv(xi_21) * w2->value() - w1->value();

  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(T1);
  node->addChild(w1);
  node->addChild(T2);
  node->addChild(w2);
  return node;
}

// See State Estimation (2nd Ed) Section 11.1.4
void PriorFactor::backward(const Eigen::MatrixXd& lhs,
                           const Node<OutType>::Ptr& node,
                           Jacobians& jacs) const {
  // e \approx ebar + JAC * delta_x
  if (knot1_->pose()->active() || knot1_->velocity()->active()) {
    const auto Fk1 = getJacKnot1(knot1_, knot2_);
    if (knot1_->pose()->active()) {
      const auto T1 = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
      Eigen::MatrixXd new_lhs = lhs * Fk1.block<6, 3>(0, 0);
      knot1_->pose()->backward(new_lhs, T1, jacs);
    }
    if (knot1_->velocity()->active()) {
      const auto w1 = std::static_pointer_cast<Node<InVelType>>(node->at(1));
      Eigen::MatrixXd new_lhs = lhs * Fk1.block<6, 3>(0, 3);
      knot1_->velocity()->backward(new_lhs, w1, jacs);
    }
  }
  if (knot2_->pose()->active() || knot2_->velocity()->active()) {
    const auto Ek = getJacKnot2(knot1_, knot2_);
    if (knot2_->pose()->active()) {
      const auto T2 = std::static_pointer_cast<Node<InPoseType>>(node->at(2));
      Eigen::MatrixXd new_lhs = lhs * Ek.block<6, 3>(0, 0);
      knot2_->pose()->backward(new_lhs, T2, jacs);
    }
    if (knot2_->velocity()->active()) {
      const auto w2 = std::static_pointer_cast<Node<InVelType>>(node->at(3));
      Eigen::MatrixXd new_lhs = lhs * Ek.block<6, 3>(0, 3);
      knot2_->velocity()->backward(new_lhs, w2, jacs);
    }
  }
}

}  // namespace const_vel_2d
}  // namespace traj
}  // namespace steam