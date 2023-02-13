#include "steam/trajectory/singer/prior_factor.hpp"
#include "steam/trajectory/singer/helper.hpp"

// // #include "steam/evaluable/se3/evaluables.hpp"
// // #include "steam/evaluable/vspace/evaluables.hpp"
// // #include "steam/trajectory/const_acc/evaluable/compose_curlyhat_evaluator.hpp"
// // #include "steam/trajectory/const_vel/evaluable/jinv_velocity_evaluator.hpp"

namespace steam {
namespace traj {
namespace singer {

// Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot1_() const {
//   return getJacKnot1(knot1_, knot2_, alpha_diag_);
// }

// auto PriorFactor::MakeShared(const Variable::ConstPtr& knot1,
//                              const Variable::ConstPtr& knot2,
//                              const Eigen::Matrix<double, 6, 1>& ad) -> Ptr {
//   return std::make_shared<PriorFactor>(knot1, knot2, ad);
// }

// PriorFactor::PriorFactor(const Variable::ConstPtr& knot1,
//                          const Variable::ConstPtr& knot2,
//                          const Eigen::Matrix<double, 6, 1>& ad)
//     : knot1_(knot1), knot2_(knot2) {
//   const double dt = (knot2_->time() - knot1_->time()).seconds();
//   Phi_ = getTran(dt, ad);
// }

// // bool PriorFactor::active() const {
// //   return knot1_->pose()->active() || knot1_->velocity()->active() ||
// //          knot1_->acceleration()->active() || knot2_->pose()->active() ||
// //          knot2_->velocity()->active() || knot2_->acceleration()->active();
// // }

// // void PriorFactor::getRelatedVarKeys(KeySet& keys) const {
// //   knot1_->pose()->getRelatedVarKeys(keys);
// //   knot1_->velocity()->getRelatedVarKeys(keys);
// //   knot1_->acceleration()->getRelatedVarKeys(keys);
// //   knot2_->pose()->getRelatedVarKeys(keys);
// //   knot2_->velocity()->getRelatedVarKeys(keys);
// //   knot2_->acceleration()->getRelatedVarKeys(keys);
// // }

// // auto PriorFactor::value() const -> OutType {
// //   //
// //   OutType error = OutType::Zero();
// //   error.block<6, 1>(0, 0) = ep_->value();
// //   error.block<6, 1>(6, 0) = ev_->value();
// //   error.block<6, 1>(12, 0) = ea_->value();
// //   return error;
// // }

// // auto PriorFactor::forward() const -> Node<OutType>::Ptr {
// //   //
// //   const auto ep = ep_->forward();
// //   const auto ev = ev_->forward();
// //   const auto ea = ea_->forward();

// //   //
// //   OutType error = OutType::Zero();
// //   error.block<6, 1>(0, 0) = ep_->value();
// //   error.block<6, 1>(6, 0) = ev_->value();
// //   error.block<6, 1>(12, 0) = ea_->value();

// //   //
// //   const auto node = Node<OutType>::MakeShared(error);
// //   node->addChild(ep);
// //   node->addChild(ev);
// //   node->addChild(ea);

// //   return node;
// // }

// // void PriorFactor::backward(const Eigen::MatrixXd& lhs,
// //                            const Node<OutType>::Ptr& node,
// //                            Jacobians& jacs) const {
// //   using OutT = Eigen::Matrix<double, 6, 1>;
// //   if (ep_->active()) {
// //     const auto ep = std::static_pointer_cast<Node<OutT>>(node->at(0));
// //     ep_->backward(lhs.leftCols(6), ep, jacs);
// //   }

// //   if (ev_->active()) {
// //     const auto ev = std::static_pointer_cast<Node<OutT>>(node->at(1));
// //     ev_->backward(lhs.middleCols(6, 6), ev, jacs);
// //   }

// //   if (ea_->active()) {
// //     const auto ea = std::static_pointer_cast<Node<OutT>>(node->at(2));
// //     ea_->backward(lhs.rightCols(6), ea, jacs);
// //   }
// // }

// Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot1_(
//   const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
//   return getJacKnot1(knot1, knot2);
// }

// Eigen::Matrix<double, 18, 18> PriorFactor::getJacKnot2_(
//   const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) {
//   return getJacKnot2(knot1, knot2);
// }

}  // namespace singer
}  // namespace traj
}  // namespace steam