#include "steam/trajectory/const_vel_se2/velocity_interpolator.hpp"

#include <iostream>
#include "steam/evaluable/se2/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/trajectory/const_vel_se2/evaluable/j_velocity_evaluator.hpp"
#include "steam/trajectory/const_vel_se2/evaluable/jinv_velocity_evaluator.hpp"
#include "steam/trajectory/const_vel_se2/helper.hpp"

namespace steam {
namespace traj {
namespace const_vel_se2 {

VelocityInterpolator::Ptr VelocityInterpolator::MakeShared(
    const Time time, const Variable::ConstPtr& knot1,
    const Variable::ConstPtr& knot2) {
  return std::make_shared<VelocityInterpolator>(time, knot1, knot2);
}

VelocityInterpolator::VelocityInterpolator(const Time time,
                                           const Variable::ConstPtr& knot1,
                                           const Variable::ConstPtr& knot2)
    : knot1_(knot1), knot2_(knot2) {
  // Calculate time constants
  const double T = (knot2->time() - knot1->time()).seconds();
  const double tau = (time - knot1->time()).seconds();
  const double kappa = (knot2->time() - time).seconds();
  const Eigen::Matrix<double, 3, 1> ones = Eigen::Matrix<double, 3, 1>::Ones();
  const auto Q_tau = getQ(tau, ones);
  const auto Qinv_T = getQinv(T, ones);
  const auto Tran_kappa = getTran(kappa);
  const auto Tran_tau = getTran(tau);
  const auto Tran_T = getTran(T);
  // Calculate interpolation values
  const auto psi = (Q_tau * Tran_kappa.transpose() * Qinv_T);
  const auto lambda = (Tran_tau - psi * Tran_T);

  // // Calculate 'psi' interpolation values
  psi11_ = psi(0, 0);
  psi12_ = psi(0, 3);
  psi21_ = psi(3, 0);
  psi22_ = psi(3, 3);
  lambda11_ = lambda(0, 0);
  lambda12_ = lambda(0, 3);
  lambda21_ = lambda(3, 0);
  lambda22_ = lambda(3, 3);
}

bool VelocityInterpolator::active() const {
  return knot1_->pose()->active() || knot1_->velocity()->active() ||
         knot2_->pose()->active() || knot2_->velocity()->active();
}

void VelocityInterpolator::getRelatedVarKeys(KeySet& keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
}

auto VelocityInterpolator::value() const -> OutType {
  const auto T1 = knot1_->pose()->value();
  const auto w1 = knot1_->velocity()->value();
  const auto T2 = knot2_->pose()->value();
  const auto w2 = knot2_->velocity()->value();
  // Get se2 algebra of relative matrix
  const auto xi_21 = (T2 / T1).vec();
  // Calculate the 6x6 associated Jacobian
  const Eigen::Matrix<double, 3, 3> J_21_inv = lgmath::se2::vec2jacinv(xi_21);
  // Calculate interpolated relative se2 algebra
  const Eigen::Matrix<double, 3, 1> xi_i1 =
      lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_21_inv * w2;
  const Eigen::Matrix<double, 3, 3> J_i1 = lgmath::se2::vec2jac(xi_i1);
  const Eigen::Matrix<double, 3, 1> xi_j1 =
      lambda22_ * w1 + psi21_ * xi_21 + psi22_ * J_21_inv * w2;
  // Calculate interpolated body-centric velocity
  OutType xi_it = J_i1 * xi_j1;
  return xi_it;
}

auto VelocityInterpolator::forward() const -> Node<OutType>::Ptr {
  const auto T1 = knot1_->pose()->forward();
  const auto w1 = knot1_->velocity()->forward();
  const auto T2 = knot2_->pose()->forward();
  const auto w2 = knot2_->velocity()->forward();
  // Get se2 algebra of relative matrix
  const auto xi_21 = (T2->value() / T1->value()).vec();
  // Calculate the 6x6 associated Jacobian
  const Eigen::Matrix<double, 3, 3> J_21_inv = lgmath::se2::vec2jacinv(xi_21);
  // Calculate interpolated relative se2 algebra
  const Eigen::Matrix<double, 3, 1> xi_i1 = lambda12_ * w1->value() +
                                            psi11_ * xi_21 +
                                            psi12_ * J_21_inv * w2->value();
  const Eigen::Matrix<double, 3, 3> J_i1 = lgmath::se2::vec2jac(xi_i1);
  const Eigen::Matrix<double, 3, 1> xi_j1 = lambda22_ * w1->value() +
                                            psi21_ * xi_21 +
                                            psi22_ * J_21_inv * w2->value();
  // Calculate interpolated body-centric velocity
  OutType xi_it = J_i1 * xi_j1;
  const auto node = Node<OutType>::MakeShared(xi_it);
  node->addChild(T1);
  node->addChild(w1);
  node->addChild(T2);
  node->addChild(w2);
  return node;
}

void VelocityInterpolator::backward(const Eigen::MatrixXd& lhs,
                                    const Node<OutType>::Ptr& node,
                                    Jacobians& jacs) const {
  if (!active()) return;
  const auto T1 = knot1_->pose()->value();
  const auto w1 = knot1_->velocity()->value();
  const auto T2 = knot2_->pose()->value();
  const auto w2 = knot2_->velocity()->value();
  // Get se2 algebra of relative matrix
  const auto xi_21 = (T2 / T1).vec();
  // Calculate the 6x6 associated Jacobian
  const Eigen::Matrix<double, 3, 3> J_21_inv = lgmath::se2::vec2jacinv(xi_21);
  // Calculate interpolated relative se2 algebra
  const Eigen::Matrix<double, 3, 1> xi_i1 =
      lambda12_ * w1 + psi11_ * xi_21 + psi12_ * J_21_inv * w2;
  const Eigen::Matrix<double, 3, 1> xi_j1 =
      lambda22_ * w1 + psi21_ * xi_21 + psi22_ * J_21_inv * w2;
  // Calculate the 3x3 Jacobian associated with the interpolated relative
  // transformation matrix
  const Eigen::Matrix<double, 3, 3> J_i1 = lgmath::se2::vec2jac(xi_i1);
  // temp value for speed
  const Eigen::Matrix<double, 3, 3> xi_j1_ch =
      -0.5 * lgmath::se2::curlyhat(xi_j1);

  // Calculate relative transformation matrix
  // const lgmath::se2::Transformation T_21(xi_21);
  const lgmath::se2::Transformation T_21 = T2 * T1.inverse();
  if (knot1_->pose()->active() || knot2_->pose()->active()) {
    // Precompute part of jacobian matrices
    const Eigen::Matrix<double, 3, 3> wtmp =
        (0.5 * lgmath::se2::curlyhat(w2)) * J_21_inv;
    const Eigen::Matrix<double, 3, 3> w =
        J_i1 * (psi21_ * J_21_inv + psi22_ * wtmp) +
        xi_j1_ch * (psi11_ * J_21_inv + psi12_ * wtmp);
    if (knot1_->pose()->active()) {
      const auto T1_ = std::static_pointer_cast<Node<InPoseType>>(node->at(0));
      Eigen::MatrixXd new_lhs = lhs * (-w * T_21.adjoint());
      knot1_->pose()->backward(new_lhs, T1_, jacs);
    }
    if (knot2_->pose()->active()) {
      const auto T2_ = std::static_pointer_cast<Node<InPoseType>>(node->at(2));
      Eigen::MatrixXd new_lhs = lhs * w;
      knot2_->pose()->backward(new_lhs, T2_, jacs);
    }
  }
  if (knot1_->velocity()->active()) {
    const auto w1_ = std::static_pointer_cast<Node<InVelType>>(node->at(1));
    Eigen::MatrixXd new_lhs = lhs * (J_i1 * lambda22_ + xi_j1_ch * lambda12_);
    knot1_->velocity()->backward(new_lhs, w1_, jacs);
  }
  if (knot2_->velocity()->active()) {
    const auto w2_ = std::static_pointer_cast<Node<InVelType>>(node->at(3));
    Eigen::MatrixXd new_lhs =
        lhs * (J_i1 * (psi22_ * J_21_inv) + xi_j1_ch * (psi12_ * J_21_inv));
    knot2_->velocity()->backward(new_lhs, w2_, jacs);
  }
}

}  // namespace const_vel_se2
}  // namespace traj
}  // namespace steam