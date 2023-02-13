#include "steam/trajectory/singer/interface.hpp"

// #include "steam/evaluable/se3/evaluables.hpp"
// #include "steam/evaluable/vspace/evaluables.hpp"
// #include "steam/problem/loss_func/loss_funcs.hpp"
// #include "steam/problem/noise_model/static_noise_model.hpp"

// #include "steam/trajectory/singer/helper.hpp"
// #include "steam/trajectory/singer/prior_factor.hpp"
// #include "steam/trajectory/singer/pose_interpolator.hpp"
// #include "steam/trajectory/singer/velocity_interpolator.hpp"
// #include "steam/trajectory/singer/acceleration_interpolator.hpp"
// #include "steam/trajectory/singer/pose_extrapolator.hpp"
// #include "steam/trajectory/singer/velocity_extrapolator.hpp"
// #include "steam/trajectory/singer/acceleration_extrapolator.hpp"

namespace steam {
namespace traj {
namespace singer {

// auto Interface::MakeShared(const Eigen::Matrix<double, 6, 1>& alpha_diag,
//                            const Eigen::Matrix<double, 6, 1>& Qc_diag) -> Ptr {
//   return std::make_shared<Interface>(alpha_diag, Qc_diag);
// }

// Interface::Interface(const Eigen::Matrix<double, 6, 1>& alpha_diag,
//                      const Eigen::Matrix<double, 6, 1>& Qc_diag)
//     : alpha_diag_(alpha_diag), Qc_diag_(Qc_diag) {}

// void Interface::add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
//                     const Evaluable<VelocityType>::Ptr& w_0k_ink,
//                     const Evaluable<VelocityType>::Ptr& dw_0k_ink) {
//   if (knot_map_.find(time) != knot_map_.end())
//     throw std::runtime_error("adding knot at duplicated time");
//   const auto knot = std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink);
//   knot_map_.insert(knot_map_.end(), std::pair<Time, Variable::Ptr>(time, knot));
// }

// auto Interface::getPoseInterpolator(const Time& time) const
//     -> Evaluable<PoseType>::ConstPtr {
//   // Check that map is not empty
//   if (knot_map_.empty()) throw std::runtime_error("knot map is empty");

//   // Get iterator to first element with time equal to or greater than 'time'
//   auto it1 = knot_map_.lower_bound(time);

//   // Check if time is passed the last entry
//   if (it1 == knot_map_.end()) {
//     --it1;  // should be safe, as we checked that the map was not empty..
//     const auto& endKnot = it1->second;
//     const auto T_t_k = PoseExtrapolator::MakeShared(time, endKnot->velocity());
//     return se3::compose(T_t_k, endKnot->pose());
//   }

//   // Check if we requested time exactly
//   if (it1->second->time() == time) return it1->second->pose();

//   // Check if we requested before first time
//   if (it1 == knot_map_.begin()) {
//     const auto& startKnot = it1->second;
//     const auto T_t_k = PoseExtrapolator::MakeShared(time, startKnot->velocity());
//     return se3::compose(T_t_k, startKnot->pose());
//   }

//   // Get iterators bounding the time interval
//   auto it2 = it1;
//   --it1;
//   if (time <= it1->second->time() || time >= it2->second->time())
//     throw std::runtime_error("Requested interpolation at an invalid time");

//   // Create interpolated evaluator
//   return PoseInterpolator::MakeShared(time, it1->second, it2->second, alpha_diag_);
// }

// auto Interface::getVelocityInterpolator(const Time& time) const
//     -> Evaluable<VelocityType>::ConstPtr {
//   // Check that map is not empty
//   if (knot_map_.empty()) throw std::runtime_error("knot map is empty");

//   // Get iterator to first element with time equal to or greater than 'time'
//   auto it1 = knot_map_.lower_bound(time);

//   // Check if time is passed the last entry
//   if (it1 == knot_map_.end()) {
//     --it1;  // should be safe, as we checked that the map was not empty..
//     const auto& endKnot = it1->second;
//     const auto w_t_k = VelocityExtrapolator::MakeShared(time - endKnot->time(),
//         endKnot->acceleration());
//     return vspace::add<6>(w_t_k, endKnot->velocity());

//   }

//   // Check if we requested time exactly
//   if (it1->second->time() == time) return it1->second->velocity();

//   // Check if we requested before first time
//   if (it1 == knot_map_.begin()) {
//     const auto& startKnot = it1->second;
//     const auto w_t_k = VelocityExtrapolator::MakeShared(time - startKnot->time(),
//         startKnot->acceleration());
//     return vspace::add<6>(w_t_k, startKnot->velocity());
//   }

//   // Get iterators bounding the time interval
//   auto it2 = it1;
//   --it1;
//   if (time <= it1->second->time() || time >= it2->second->time())
//     throw std::runtime_error("Requested interpolation at an invalid time");

//   // Create interpolated evaluator
//   return VelocityInterpolator::MakeShared(time, it1->second, it2->second, alpha_diag_);
// }

// auto Interface::getAccelerationInterpolator(const Time& time) const
//     -> Evaluable<AccelerationType>::ConstPtr {
//   // Check that map is not empty
//   if (knot_map_.empty()) throw std::runtime_error("knot map is empty");

//   // Get iterator to first element with time equal to or greater than 'time'
//   auto it1 = knot_map_.lower_bound(time);

//   // Check if time is passed the last entry
//   if (it1 == knot_map_.end()) {
//     --it1;  // should be safe, as we checked that the map was not empty..
//     const auto& endKnot = it1->second;
//     return endKnot->acceleration();
//   }

//   // Check if we requested time exactly
//   if (it1->second->time() == time) return it1->second->acceleration();

//   // Check if we requested before first time
//   if (it1 == knot_map_.begin()) {
//     const auto& startKnot = it1->second;
//     return startKnot->acceleration();
//   }

//   // Get iterators bounding the time interval
//   auto it2 = it1;
//   --it1;
//   if (time <= it1->second->time() || time >= it2->second->time())
//     throw std::runtime_error("Requested interpolation at an invalid time");

//   // Create interpolated evaluator
//   return AccelerationInterpolator::MakeShared(time, it1->second, it2->second, alpha_diag_);
// }

// // See State Estimation (2nd Ed) Sections 11.1.4, 11.3.2
// auto Interface::getCovariance(const Covariance& cov, const Time& time)
//     -> CovType {
//   // clang-format off

//   //
//   if (knot_map_.empty()) throw std::runtime_error("map is empty");

//   // Get iterator to first element with time equal to or greater than 'time'
//   auto it1 = knot_map_.lower_bound(time);

//   // extrapolate after last entry
//   if (it1 == knot_map_.end()) {
//     --it1;  // should be safe, as we checked that the map was not empty..

//     const auto& endKnot = it1->second;
//     const auto T_k0 = endKnot->pose();
//     const auto w_0k_ink = endKnot->velocity();
//     const auto dw_0k_ink = endKnot->acceleration();
//     if (!T_k0->active() || !w_0k_ink->active() || !dw_0k_ink->active())
//       throw std::runtime_error("extrapolation from a locked knot not implemented.");

//     const auto T_k0_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_k0);
//     const auto w_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_0k_ink);
//     const auto dw_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_0k_ink);
//     if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
//       throw std::runtime_error("trajectory states are not variables.");

//     // Construct a knot for the extrapolated state
//     const auto T_t_0 = PoseExtrapolator::MakeShared(time - endKnot->time(), endKnot, alpha_diag_);
//     const auto w_t_0 = VelocityExtrapolator::MakeShared(time - endKnot->time(), endKnot, alpha_diag_);
//     const auto dw_t_0 = AccelerationExtrapolator::MakeShared(time - endKnot->time(), endKnot, alpha_diag_);
    
//     const auto extrap_knot = Variable::MakeShared(time, T_t_0, w_t_0, dw_t_0);

//     // Compute Jacobians
//     // Note: jacKnot1 will return the negative of F as defined in
//     // the state estimation textbook where we take the interpolation equations.
//     // This doesn't apply to jacKnot2.
//     const auto F_t1 = -getJacKnot1(endKnot, extrap_knot, alpha_diag_);
//     const auto E_t1_inv = getJacKnot2(endKnot, extrap_knot).inverse();

//     // Prior covariance
//     const auto Qt1 = getQ((extrap_knot->time() - endKnot->time()).seconds(), alpha_diag_, Qc_diag_);

//     // end knot covariance
//     const std::vector<StateVarBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var, dw_0k_ink_var};
//     const Eigen::Matrix<double, 18, 18> P_end = cov.query(state_var);

//     // Compute covariance
//     return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1) * E_t1_inv.transpose();
//   }

//   // Check if we requested time exactly
//   if (it1->second->time() == time) {
//     const auto& knot = it1->second;
//     const auto T_k0 = knot->pose();
//     const auto w_0k_ink = knot->velocity();
//     const auto dw_0k_ink = knot->acceleration();
//     if (!T_k0->active() || !w_0k_ink->active() || !dw_0k_ink->active())
//       throw std::runtime_error("extrapolation from a locked knot not implemented.");

//     const auto T_k0_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_k0);
//     const auto w_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_0k_ink);
//     const auto dw_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_0k_ink);
//     if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
//       throw std::runtime_error("trajectory states are not variables.");

//     std::vector<StateVarBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var, dw_0k_ink_var};
//     return cov.query(state_var);
//   }

//   // Check if we requested before first time
//   if (it1 == knot_map_.begin())
//     throw std::runtime_error("Requested covariance before first time.");


//   // Get iterators bounding the time interval
//   auto it2 = it1;
//   --it1;

//   const auto& knot1 = it1->second;
//   const auto T_10 = knot1->pose();
//   const auto w_01_in1 = knot1->velocity();
//   const auto dw_01_in1 = knot1->acceleration();
//   const auto& knot2 = it2->second;
//   const auto T_20 = knot2->pose();
//   const auto w_02_in2 = knot2->velocity();
//   const auto dw_02_in2 = knot2->acceleration();
//   if (!T_10->active() || !w_01_in1->active() || !dw_01_in1->active()
//     || !T_20->active() || !w_02_in2->active() || !dw_02_in2->active())
//     throw std::runtime_error("extrapolation from a locked knot not implemented.");

//   const auto T_10_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_10);
//   const auto w_01_in1_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_01_in1);
//   const auto dw_01_in1_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_01_in1);
//   const auto T_20_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_20);
//   const auto w_02_in2_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_02_in2);
//   const auto dw_02_in2_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_02_in2);
//   if (!T_10_var || !w_01_in1_var || !dw_01_in1_var || !T_20_var || !w_02_in2_var || !dw_02_in2_var)
//     throw std::runtime_error("trajectory states are not variables.");

//   // Construct a knot for the interpolated state
//   const auto T_q0_eval = PoseInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
//   const auto w_0q_inq_eval = VelocityInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
//   const auto dw_0q_inq_eval = AccelerationInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
//   const auto knotq = Variable::MakeShared(time, T_q0_eval, w_0q_inq_eval, dw_0q_inq_eval);

//   // Compute Jacobians
//   // Note: jacKnot1 will return the negative of F as defined in
//   // the state estimation textbook where we take the interpolation equations.
//   // This doesn't apply to jacKnot2.
//   const Eigen::Matrix<double, 18, 18> F_t1 = -getJacKnot1(knot1, knotq);
//   const Eigen::Matrix<double, 18, 18> E_t1 = getJacKnot2(knot1, knotq);
//   const Eigen::Matrix<double, 18, 18> F_2t = -getJacKnot1(knotq, knot2);
//   const Eigen::Matrix<double, 18, 18> E_2t = getJacKnot2(knotq, knot2);

//   // Prior inverse covariances
//   const Eigen::Matrix<double, 18, 18> Qt1_inv = getQ((knotq->time() - knot1->time()).seconds(), alpha_diag_, Qc_diag_).inverse();
//   const Eigen::Matrix<double, 18, 18> Q2t_inv = getQ((knot2->time() - knotq->time()).seconds(), alpha_diag_, Qc_diag_).inverse();

//   // Covariance of knot1 and knot2
//   const std::vector<StateVarBase::ConstPtr> state_var{T_10_var, w_01_in1_var, T_20_var, w_02_in2_var};
//   const Eigen::Matrix<double, 36, 36> P_1n2 = cov.query(state_var);

//   // Helper matrices
//   Eigen::Matrix<double, 36, 18> A = Eigen::Matrix<double, 36, 18>::Zero();
//   A.block<18, 18>(0, 0) = F_t1.transpose() * Qt1_inv * E_t1;
//   A.block<18, 18>(18, 0) = E_2t.transpose() * Q2t_inv * F_2t;

//   Eigen::Matrix<double, 36, 36> B = Eigen::Matrix<double, 36, 36>::Zero();
//   B.block<18, 18>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
//   B.block<18, 18>(18, 18) = E_2t.transpose() * Q2t_inv * E_2t;

//   const Eigen::Matrix<double, 18, 18> F_21 = -getJacKnot1(knot1, knot2);
//   const Eigen::Matrix<double, 18, 18> E_21 = getJacKnot2(knot1, knot2);
//   const Eigen::Matrix<double, 18, 18> Q21_inv = getQ((knot2->time() - knot1->time()).seconds(), alpha_diag_, Qc_diag_).inverse();

//   Eigen::Matrix<double, 36, 36> Pinv_comp = Eigen::Matrix<double, 36, 36>::Zero();
//   Pinv_comp.block<18, 18>(0, 0) = F_21.transpose() * Q21_inv * F_21;
//   Pinv_comp.block<18, 18>(18, 0) = -E_21.transpose() * Q21_inv * F_21;
//   Pinv_comp.block<18, 18>(0, 18) = Pinv_comp.block<18, 18>(18, 0).transpose();
//   Pinv_comp.block<18, 18>(18, 18) = E_21.transpose() * Q21_inv * E_21;

//   // interpolated covariance
//   const Eigen::Matrix<double, 18, 18> P_t_inv = E_t1.transpose() * Qt1_inv * E_t1 + F_2t.transpose() * Q2t_inv * F_2t -
//                  A.transpose() * (P_1n2.inverse() + B - Pinv_comp).inverse() * A;

//   return P_t_inv.inverse();
//   // clang-format on
// }

// void Interface::addPosePrior(const Time& time, const PoseType& T_k0,
//                              const Eigen::Matrix<double, 6, 6>& cov) {
//   if (pose_prior_factor_ != nullptr)
//     throw std::runtime_error("can only add one pose prior.");

//   // Check that map is not empty
//   if (knot_map_.empty()) throw std::runtime_error("knot map is empty.");

//   // Try to find knot at same time
//   auto it = knot_map_.find(time);
//   if (it == knot_map_.end())
//     throw std::runtime_error("no knot at provided time.");

//   // Get reference
//   const auto& knot = it->second;

//   // Check that the pose is not locked
//   if (!knot->pose()->active())
//     throw std::runtime_error("tried to add prior to locked pose.");

//   // Set up loss function, noise model, and error function
//   auto error_func = se3::se3_error(knot->pose(), T_k0);
//   auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
//   auto loss_func = L2LossFunc::MakeShared();

//   // Create cost term
//   pose_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
//       error_func, noise_model, loss_func);
// }

// void Interface::addVelocityPrior(const Time& time, const VelocityType& w_0k_ink,
//                                  const Eigen::Matrix<double, 6, 6>& cov) {
//   if (vel_prior_factor_ != nullptr)
//     throw std::runtime_error("can only add one velocity prior.");

//   // Check that map is not empty
//   if (knot_map_.empty()) throw std::runtime_error("knot map is empty.");

//   // Try to find knot at same time
//   auto it = knot_map_.find(time);
//   if (it == knot_map_.end())
//     throw std::runtime_error("no knot at provided time.");

//   // Get reference
//   const auto& knot = it->second;

//   // Check that the velocity is not locked
//   if (!knot->velocity()->active())
//     throw std::runtime_error("tried to add prior to locked velocity.");

//   // Set up loss function, noise model, and error function
//   auto error_func = vspace::vspace_error<6>(knot->velocity(), w_0k_ink);
//   auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
//   auto loss_func = L2LossFunc::MakeShared();

//   // Create cost term
//   vel_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
//       error_func, noise_model, loss_func);
// }

// void Interface::addAccelerationPrior(const Time& time,
//                                      const AccelerationType& dw_0k_ink,
//                                      const Eigen::Matrix<double, 6, 6>& cov) {
//   if (acc_prior_factor_ != nullptr)
//     throw std::runtime_error("can only add one acceleration prior.");

//   // Check that map is not empty
//   if (knot_map_.empty()) throw std::runtime_error("knot map is empty.");

//   // Try to find knot at same time
//   auto it = knot_map_.find(time);
//   if (it == knot_map_.end())
//     throw std::runtime_error("no knot at provided time.");

//   // Get reference
//   const auto& knot = it->second;

//   // Check that the acceleration is not locked
//   if (!knot->acceleration()->active())
//     throw std::runtime_error("tried to add prior to locked acceleration.");

//   // Set up loss function, noise model, and error function
//   auto error_func = vspace::vspace_error<6>(knot->acceleration(), dw_0k_ink);
//   auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
//   auto loss_func = L2LossFunc::MakeShared();

//   // Create cost term
//   acc_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
//       error_func, noise_model, loss_func);
// }

// void Interface::addPriorCostTerms(Problem& problem) const {
//   // If empty, return none
//   if (knot_map_.empty()) return;

//   // Check for pose or velocity priors
//   if (pose_prior_factor_ != nullptr) problem.addCostTerm(pose_prior_factor_);
//   if (vel_prior_factor_ != nullptr) problem.addCostTerm(vel_prior_factor_);
//   if (acc_prior_factor_ != nullptr) problem.addCostTerm(acc_prior_factor_);

//   // All prior factors will use an L2 loss function
//   const auto loss_function = std::make_shared<L2LossFunc>();

//   // Initialize iterators
//   auto it1 = knot_map_.begin();
//   auto it2 = it1;
//   ++it2;

//   // Iterate through all states.. if any are unlocked, supply a prior term
//   for (; it2 != knot_map_.end(); ++it1, ++it2) {
//     // Get knots
//     const auto& knot1 = it1->second;
//     const auto& knot2 = it2->second;

//     // Check if any of the variables are unlocked
//     if (knot1->pose()->active() || knot1->velocity()->active() ||
//         knot1->acceleration()->active() || knot2->pose()->active() ||
//         knot2->velocity()->active() || knot2->acceleration()->active()) {
//       // Generate information matrix for GP prior factor
//       auto Qinv = getQ((knot2->time() - knot1->time()).seconds(), alpha_diag_, Qc_diag_).inverse();
//       const auto noise_model =
//           std::make_shared<StaticNoiseModel<18>>(Qinv, NoiseType::COVARIANCE);
//       //
//       const auto error_function =
//           PriorFactor::MakeShared(knot1, knot2, alpha_diag_);
//       // Create cost term
//       const auto cost_term = std::make_shared<WeightedLeastSqCostTerm<18>>(
//           error_function, noise_model, loss_function);
//       //
//       problem.addCostTerm(cost_term);
//     }
//   }
// }

// Eigen::Matrix<double, 18, 18> Interface::getJacKnot1_(
//   const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
//   return getJacKnot1(knot1, knot2, alpha_diag_);
// }

// Eigen::Matrix<double, 18, 18> Interface::getQ_(
//   const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
//   return getQ(dt, alpha_diag_, Qc_diag);
// }

// Eigen::Matrix<double, 18, 18> Interface::getQinv_(
//   const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
//   return getQ(dt, alpha_diag_, Qc_diag).inverse();
// }

// Evaluable<PoseType>::ConstPtr Interface::getPoseInterpolator_(const Time& time,
//   const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
//   PoseInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
// }

// Evaluable<PoseType>::ConstPtr Interface::getVelocityInterpolator_(const Time& time,
//   const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
//   VelocityInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
// }

// Evaluable<PoseType>::ConstPtr Interface::getAccelerationInterpolator_(const Time& time,
//   const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
//   AccelerationInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
// }

// Evaluable<PoseType>::ConstPtr Interface::getPoseExtrapolator_(const Time& time,
//   const Variable::ConstPtr& knot) const {
//   PoseExtrapolator::MakeShared(time, knot, alpha_diag_);
// }

// Evaluable<PoseType>::ConstPtr Interface::getVelocityExtrapolator_(const Time& time,
//   const Variable::ConstPtr& knot) const {
//   VelocityExtrapolator::MakeShared(time, knot, alpha_diag_);
// }

// Evaluable<PoseType>::ConstPtr Interface::getAccelerationExtrapolator_(const Time& time,
//   const Variable::ConstPtr& knot) const {
//   AccelerationExtrapolator::MakeShared(time, knot, alpha_diag_);
// }

// Evaluable<Eigen::Matrix<double, 18, 1>>::ConstPtr Interface::getPriorFactor_(
//   const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
//   PriorFactor::MakeShared(knot1, knot2, alpha_diag_);
// }


}  // namespace singer
}  // namespace traj
}  // namespace steam