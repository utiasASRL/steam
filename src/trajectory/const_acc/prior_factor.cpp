#include "steam/trajectory/const_acc/prior_factor.hpp"
#include "steam/trajectory/const_acc/helper.hpp"

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
  const double dt = (knot2_->time() - knot1_->time()).seconds();
  Phi_ = getTran(dt);
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
  OutType error = OutType::Zero();

  const auto T1 = knot1_->pose()->value();
  const auto w1 = knot1_->velocity()->value();
  const auto dw1 = knot1_->acceleration()->value();
  const auto T2 = knot2_->pose()->value();
  const auto w2 = knot2_->velocity()->value();
  const auto dw2 = knot2_->acceleration()->value();

  const auto xi_21 = (T2 / T1).vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  Eigen::Matrix<double, 18, 1> gamma1 = Eigen::Matrix<double, 18, 1>::Zero();
  gamma1.block<6, 1>(6, 0) = w1;
  gamma1.block<6, 1>(12, 0) = dw1;
  Eigen::Matrix<double, 18, 1> gamma2 = Eigen::Matrix<double, 18, 1>::Zero();
  gamma2.block<6, 1>(0, 0) = xi_21;
  gamma2.block<6, 1>(6, 0) = J_21_inv * w2;
  gamma2.block<6, 1>(12, 0) =
      -0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2;
  error = gamma2 - Phi_ * gamma1;
  return error;
}

auto PriorFactor::forward() const -> Node<OutType>::Ptr {
  const auto T1 = knot1_->pose()->forward();
  const auto w1 = knot1_->velocity()->forward();
  const auto dw1 = knot1_->acceleration()->forward();
  const auto T2 = knot2_->pose()->forward();
  const auto w2 = knot2_->velocity()->forward();
  const auto dw2 = knot2_->acceleration()->forward();

  const auto xi_21 = (T2->value() / T1->value()).vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  OutType error = OutType::Zero();
  Eigen::Matrix<double, 18, 1> gamma1 = Eigen::Matrix<double, 18, 1>::Zero();
  gamma1.block<6, 1>(6, 0) = w1->value();
  gamma1.block<6, 1>(12, 0) = dw1->value();
  Eigen::Matrix<double, 18, 1> gamma2 = Eigen::Matrix<double, 18, 1>::Zero();
  gamma2.block<6, 1>(0, 0) = xi_21;
  gamma2.block<6, 1>(6, 0) = J_21_inv * w2->value();
  gamma2.block<6, 1>(12, 0) =
      -0.5 * lgmath::se3::curlyhat(J_21_inv * w2->value()) * w2->value() +
      J_21_inv * dw2->value();
  error = gamma2 - Phi_ * gamma1;
  const auto node = Node<OutType>::MakeShared(error);
  node->addChild(T1);
  node->addChild(w1);
  node->addChild(dw1);
  node->addChild(T2);
  node->addChild(w2);
  node->addChild(dw2);
  return node;
}

// See State Estimation (2nd Ed) Section 11.1.4
void PriorFactor::backward(const Eigen::MatrixXd& lhs,
                           const Node<OutType>::Ptr& node,
                           Jacobians& jacs) const {
  // e \approx ebar + JAC * delta_x
  if (knot1_->pose()->active() || knot1_->velocity()->active() ||
      knot1_->acceleration()->active()) {
    const auto Fk1 = getJacKnot1_();
    if (knot1_->pose()->active()) {
      const auto T1 = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
      Eigen::MatrixXd new_lhs = lhs * Fk1.block<18, 6>(0, 0);
      knot1_->pose()->backward(new_lhs, T1, jacs);
    }
    if (knot1_->velocity()->active()) {
      const auto w1 = std::static_pointer_cast<Node<InVelType>>(node->at(1));
      Eigen::MatrixXd new_lhs = lhs * Fk1.block<18, 6>(0, 6);
      knot1_->velocity()->backward(new_lhs, w1, jacs);
    }
    if (knot1_->acceleration()->active()) {
      const auto dw1 = std::static_pointer_cast<Node<InAccType>>(node->at(2));
      Eigen::MatrixXd new_lhs = lhs * Fk1.block<18, 6>(0, 12);
      knot1_->acceleration()->backward(new_lhs, dw1, jacs);
    }
  }
  if (knot2_->pose()->active() || knot2_->velocity()->active() ||
      knot2_->acceleration()->active()) {
    const auto Ek = getJacKnot2_();
    if (knot2_->pose()->active()) {
      const auto T2 = std::static_pointer_cast<Node<InPoseType>>(node->at(3));
      Eigen::MatrixXd new_lhs = lhs * Ek.block<18, 6>(0, 0);
      knot2_->pose()->backward(new_lhs, T2, jacs);
    }
    if (knot2_->velocity()->active()) {
      const auto w2 = std::static_pointer_cast<Node<InVelType>>(node->at(4));
      Eigen::MatrixXd new_lhs = lhs * Ek.block<18, 6>(0, 6);
      knot2_->velocity()->backward(new_lhs, w2, jacs);
    }
    if (knot2_->acceleration()->active()) {
      const auto dw2 = std::static_pointer_cast<Node<InAccType>>(node->at(5));
      Eigen::MatrixXd new_lhs = lhs * Ek.block<18, 6>(0, 12);
      knot2_->acceleration()->backward(new_lhs, dw2, jacs);
    }
  }
}

Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot1_() const {
  return getJacKnot1(knot1_, knot2_);
}

Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot2_() const {
  return getJacKnot2(knot1_, knot2_);
}

}  // namespace const_acc
}  // namespace traj
}  // namespace steam
