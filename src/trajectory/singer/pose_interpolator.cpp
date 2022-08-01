#include "steam/trajectory/singer/pose_interpolator.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/trajectory/const_acc/evaluable/compose_curlyhat_evaluator.hpp"
#include "steam/trajectory/const_vel/evaluable/jinv_velocity_evaluator.hpp"
#include "steam/trajectory/singer/helper.hpp"

namespace steam {
namespace traj {
namespace singer {

PoseInterpolator::Ptr PoseInterpolator::MakeShared(
    const Time& time, const Variable::ConstPtr& knot1,
    const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad) {
  return std::make_shared<PoseInterpolator>(time, knot1, knot2, ad);
}

PoseInterpolator::PoseInterpolator(const Time& time,
                                   const Variable::ConstPtr& knot1,
                                   const Variable::ConstPtr& knot2,
                                   const Eigen::Matrix<double, 6, 1>& ad)
    : knot1_(knot1), knot2_(knot2) {
  // Calculate time constants
  const double T = (knot2->time() - knot1->time()).seconds();
  const double tau = (time - knot1->time()).seconds();
  const double kappa = (knot2->time() - time).seconds();

  // Q and Transition matrix
  const auto Q_tau = getQ(tau, ad);
  const auto Q_T = getQ(T, ad);
  const auto Tran_kappa = getTran(kappa, ad);
  const auto Tran_tau = getTran(tau, ad);
  const auto Tran_T = getTran(T, ad);

  // Calculate interpolation values
  Eigen::Matrix<double, 18, 18> Omega(Q_tau * Tran_kappa.transpose() *
                                      Q_T.inverse());
  Eigen::Matrix<double, 18, 18> Lambda(Tran_tau - Omega * Tran_T);

  // construct computation graph
  const auto T1 = knot1_->pose();
  const auto w1 = knot1_->velocity();
  const auto dw1 = knot1_->acceleration();
  const auto T2 = knot2_->pose();
  const auto w2 = knot2_->velocity();
  const auto dw2 = knot2_->acceleration();

  using namespace steam::se3;
  using namespace steam::vspace;

  // clang-format off

  // Get relative matrix info
  const auto T_21 = compose_rinv(T2, T1);
  // Get se3 algebra of relative matrix
  const auto xi_21 = tran2vec(T_21);
  //
  const auto gamma11 = w1;
  const auto gamma12 = dw1;
  const auto gamma20 = xi_21;
  const auto gamma21 = const_vel::jinv_velocity(xi_21, w2);
  const auto gamma22 = add<6>(smult<6>(const_acc::compose_curlyhat(const_vel::jinv_velocity(xi_21, w2), w2), -0.5), const_vel::jinv_velocity(xi_21, dw2));

  // pose
  const auto _t1 = mmult<6>(gamma11, Lambda.block<6, 6>(0, 6));
  const auto _t2 = mmult<6>(gamma12, Lambda.block<6, 6>(0, 12));
  const auto _t3 = mmult<6>(gamma20, Omega.block<6, 6>(0, 0));
  const auto _t4 = mmult<6>(gamma21, Omega.block<6, 6>(0, 6));
  const auto _t5 = mmult<6>(gamma22, Omega.block<6, 6>(0, 12));
  const auto xi_i1 = add<6>(_t1, add<6>(_t2, add<6>(_t3, add<6>(_t4, _t5))));

  // calculate interpolated relative transformation matrix
  const auto T_i1 = vec2tran(xi_i1);
  // compose to get global transform
  T_i0_ = compose(T_i1, T1);

  // clang-format on
}

bool PoseInterpolator::active() const {
  return knot1_->pose()->active() || knot1_->velocity()->active() ||
         knot1_->acceleration()->active() || knot2_->pose()->active() ||
         knot2_->velocity()->active() || knot2_->acceleration()->active();
}

void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot1_->acceleration()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
  knot2_->acceleration()->getRelatedVarKeys(keys);
}

auto PoseInterpolator::value() const -> OutType { return T_i0_->value(); }

auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
  return T_i0_->forward();
}

void PoseInterpolator::backward(const Eigen::MatrixXd& lhs,
                                const Node<OutType>::Ptr& node,
                                Jacobians& jacs) const {
  return T_i0_->backward(lhs, node, jacs);
}

}  // namespace singer
}  // namespace traj
}  // namespace steam