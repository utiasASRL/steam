// #include "steam/trajectory/singer/pose_interpolator.hpp"

// #include "steam/evaluable/se3/evaluables.hpp"
// #include "steam/evaluable/vspace/evaluables.hpp"
// #include "steam/trajectory/const_acc/evaluable/compose_curlyhat_evaluator.hpp"
// #include "steam/trajectory/const_vel/evaluable/jinv_velocity_evaluator.hpp"
// #include "steam/trajectory/singer/helper.hpp"

// namespace steam {
// namespace traj {
// namespace singer {

// PoseInterpolator::Ptr PoseInterpolator::MakeShared(
//     const Time& time, const Variable::ConstPtr& knot1,
//     const Variable::ConstPtr& knot2, const Eigen::Matrix<double, 6, 1>& ad) {
//   return std::make_shared<PoseInterpolator>(time, knot1, knot2, ad);
// }

// PoseInterpolator::PoseInterpolator(const Time& time,
//                                    const Variable::ConstPtr& knot1,
//                                    const Variable::ConstPtr& knot2,
//                                    const Eigen::Matrix<double, 6, 1>& ad)
//     : knot1_(knot1), knot2_(knot2) {
//   // Calculate time constants
//   const double T = (knot2->time() - knot1->time()).seconds();
//   const double tau = (time - knot1->time()).seconds();
//   const double kappa = (knot2->time() - time).seconds();

//   // Q and Transition matrix
//   const auto Q_tau = getQ(tau, ad);
//   const auto Q_T = getQ(T, ad);
//   const auto Tran_kappa = getTran(kappa, ad);
//   const auto Tran_tau = getTran(tau, ad);
//   const auto Tran_T = getTran(T, ad);

//   // Calculate interpolation values
//   omega_ = Q_tau * Tran_kappa.transpose() * Q_T.inverse();
//   lambda_ = Tran_tau - Omega * Tran_T;
// }

// // bool PoseInterpolator::active() const {
// //   return knot1_->pose()->active() || knot1_->velocity()->active() ||
// //          knot1_->acceleration()->active() || knot2_->pose()->active() ||
// //          knot2_->velocity()->active() || knot2_->acceleration()->active();
// // }

// // void PoseInterpolator::getRelatedVarKeys(KeySet& keys) const {
// //   knot1_->pose()->getRelatedVarKeys(keys);
// //   knot1_->velocity()->getRelatedVarKeys(keys);
// //   knot1_->acceleration()->getRelatedVarKeys(keys);
// //   knot2_->pose()->getRelatedVarKeys(keys);
// //   knot2_->velocity()->getRelatedVarKeys(keys);
// //   knot2_->acceleration()->getRelatedVarKeys(keys);
// // }

// // auto PoseInterpolator::value() const -> OutType { return T_i0_->value(); }

// // auto PoseInterpolator::forward() const -> Node<OutType>::Ptr {
// //   return T_i0_->forward();
// // }

// // void PoseInterpolator::backward(const Eigen::MatrixXd& lhs,
// //                                 const Node<OutType>::Ptr& node,
// //                                 Jacobians& jacs) const {
// //   return T_i0_->backward(lhs, node, jacs);
// // }

// }  // namespace singer
// }  // namespace traj
// }  // namespace steam