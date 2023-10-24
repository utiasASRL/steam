#include "steam/trajectory/const_acc/interface.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/noise_model/static_noise_model.hpp"
#include "steam/trajectory/const_acc/acceleration_extrapolator.hpp"
#include "steam/trajectory/const_acc/acceleration_interpolator.hpp"
#include "steam/trajectory/const_acc/helper.hpp"
#include "steam/trajectory/const_acc/pose_extrapolator.hpp"
#include "steam/trajectory/const_acc/pose_interpolator.hpp"
#include "steam/trajectory/const_acc/prior_factor.hpp"
#include "steam/trajectory/const_acc/velocity_extrapolator.hpp"
#include "steam/trajectory/const_acc/velocity_interpolator.hpp"

namespace steam {
namespace traj {
namespace const_acc {

auto Interface::MakeShared(const Eigen::Matrix<double, 6, 1>& Qc_diag) -> Ptr {
  return std::make_shared<Interface>(Qc_diag);
}

Interface::Interface(const Eigen::Matrix<double, 6, 1>& Qc_diag)
    : Qc_diag_(Qc_diag) {}

void Interface::add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                    const Evaluable<VelocityType>::Ptr& w_0k_ink,
                    const Evaluable<VelocityType>::Ptr& dw_0k_ink) {
  if (knot_map_.find(time) != knot_map_.end())
    throw std::runtime_error("adding knot at duplicated time");
  const auto knot = std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink);
  knot_map_.insert(knot_map_.end(), std::pair<Time, Variable::Ptr>(time, knot));
}

Variable::ConstPtr Interface::get(const Time& time) const {
  return knot_map_.at(time)
}

auto Interface::getPoseInterpolator(const Time& time) const
    -> Evaluable<PoseType>::ConstPtr {
  // Check that map is not empty
  if (knot_map_.empty()) throw std::runtime_error("knot map is empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knot_map_.lower_bound(time);

  // Check if time is passed the last entry
  if (it1 == knot_map_.end()) {
    --it1;  // should be safe, as we checked that the map was not empty..
    const auto& endKnot = it1->second;
    return getPoseExtrapolator_(time, endKnot);
  }

  // Check if we requested time exactly
  if (it1->second->time() == time) return it1->second->pose();

  // Check if we requested before first time
  if (it1 == knot_map_.begin()) {
    const auto& startKnot = it1->second;
    return getPoseExtrapolator_(time, startKnot);
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;
  if (time <= it1->second->time() || time >= it2->second->time())
    throw std::runtime_error(
        "Requested interpolation at an invalid time: " +
        std::to_string(time.seconds()) + " not in (" +
        std::to_string(it1->second->time().seconds()) + ", " +
        std::to_string(it2->second->time().seconds()) + ")");

  // Create interpolated evaluator
  return getPoseInterpolator_(time, it1->second, it2->second);
}

auto Interface::getVelocityInterpolator(const Time& time) const
    -> Evaluable<VelocityType>::ConstPtr {
  // Check that map is not empty
  if (knot_map_.empty()) throw std::runtime_error("knot map is empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knot_map_.lower_bound(time);

  // Check if time is passed the last entry
  if (it1 == knot_map_.end()) {
    --it1;  // should be safe, as we checked that the map was not empty..
    const auto& endKnot = it1->second;
    return getVelocityExtrapolator_(time, endKnot);
  }

  // Check if we requested time exactly
  if (it1->second->time() == time) return it1->second->velocity();

  // Check if we requested before first time
  if (it1 == knot_map_.begin()) {
    const auto& startKnot = it1->second;
    return getVelocityExtrapolator_(time, startKnot);
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;
  if (time <= it1->second->time() || time >= it2->second->time())
    throw std::runtime_error(
        "Requested interpolation at an invalid time: " +
        std::to_string(time.seconds()) + " not in (" +
        std::to_string(it1->second->time().seconds()) + ", " +
        std::to_string(it2->second->time().seconds()) + ")");

  // Create interpolated evaluator
  return getVelocityInterpolator_(time, it1->second, it2->second);
}

auto Interface::getAccelerationInterpolator(const Time& time) const
    -> Evaluable<AccelerationType>::ConstPtr {
  // Check that map is not empty
  if (knot_map_.empty()) throw std::runtime_error("knot map is empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knot_map_.lower_bound(time);

  // Check if time is passed the last entry
  if (it1 == knot_map_.end()) {
    --it1;  // should be safe, as we checked that the map was not empty..
    const auto& endKnot = it1->second;
    return getAccelerationExtrapolator_(time, endKnot);
  }

  // Check if we requested time exactly
  if (it1->second->time() == time) return it1->second->acceleration();

  // Check if we requested before first time
  if (it1 == knot_map_.begin()) {
    const auto& startKnot = it1->second;
    return getAccelerationExtrapolator_(time, startKnot);
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;
  if (time <= it1->second->time() || time >= it2->second->time())
    throw std::runtime_error(
        "Requested interpolation at an invalid time: " +
        std::to_string(time.seconds()) + " not in (" +
        std::to_string(it1->second->time().seconds()) + ", " +
        std::to_string(it2->second->time().seconds()) + ")");

  // Create interpolated evaluator
  return getAccelerationInterpolator_(time, it1->second, it2->second);
}

// See State Estimation (2nd Ed) Sections 11.1.4, 11.3.2
auto Interface::getCovariance(const Covariance& cov, const Time& time)
    -> CovType {
  // clang-format off

  //
  if (knot_map_.empty()) throw std::runtime_error("map is empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knot_map_.lower_bound(time);

  // extrapolate after last entry
  if (it1 == knot_map_.end()) {
    --it1;  // should be safe, as we checked that the map was not empty..

    const auto& endKnot = it1->second;
    const auto T_k0 = endKnot->pose();
    const auto w_0k_ink = endKnot->velocity();
    const auto dw_0k_ink = endKnot->acceleration();
    if (!T_k0->active() || !w_0k_ink->active() || !dw_0k_ink->active())
      throw std::runtime_error("extrapolation from a locked knot not implemented.");

    const auto T_k0_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_k0);
    const auto w_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_0k_ink);
    const auto dw_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_0k_ink);
    if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
      throw std::runtime_error("trajectory states are not variables.");

    // Construct a knot for the extrapolated state
    const auto T_t_0 = getPoseExtrapolator_(time, endKnot);
    const auto w_t_0 = getVelocityExtrapolator_(time, endKnot);
    const auto dw_t_0 = getAccelerationExtrapolator_(time, endKnot);
    
    const auto extrap_knot = Variable::MakeShared(time, T_t_0, w_t_0, dw_t_0);

    // Compute Jacobians
    // Note: jacKnot1 will return the negative of F as defined in
    // the state estimation textbook where we take the interpolation equations.
    // This doesn't apply to jacKnot2.
    const auto F_t1 = -getJacKnot1_(endKnot, extrap_knot);
    const auto E_t1_inv = getJacKnot2_(endKnot, extrap_knot).inverse();

    // Prior covariance
    const auto Qt1 = getQ_((extrap_knot->time() - endKnot->time()).seconds(), Qc_diag_);

    // end knot covariance
    const std::vector<StateVarBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var, dw_0k_ink_var};
    const Eigen::Matrix<double, 18, 18> P_end = cov.query(state_var);

    // Compute covariance
    return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1) * E_t1_inv.transpose();
  }

  // Check if we requested time exactly
  if (it1->second->time() == time) {
    const auto& knot = it1->second;
    const auto T_k0 = knot->pose();
    const auto w_0k_ink = knot->velocity();
    const auto dw_0k_ink = knot->acceleration();
    if (!T_k0->active() || !w_0k_ink->active() || !dw_0k_ink->active())
      throw std::runtime_error("extrapolation from a locked knot not implemented.");

    const auto T_k0_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_k0);
    const auto w_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_0k_ink);
    const auto dw_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_0k_ink);
    if (!T_k0_var || !w_0k_ink_var || !dw_0k_ink_var)
      throw std::runtime_error("trajectory states are not variables.");

    std::vector<StateVarBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var, dw_0k_ink_var};
    return cov.query(state_var);
  }

  // Check if we requested before first time
  if (it1 == knot_map_.begin())
    throw std::runtime_error("Requested covariance before first time.");


  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;

  const auto& knot1 = it1->second;
  const auto T_10 = knot1->pose();
  const auto w_01_in1 = knot1->velocity();
  const auto dw_01_in1 = knot1->acceleration();
  const auto& knot2 = it2->second;
  const auto T_20 = knot2->pose();
  const auto w_02_in2 = knot2->velocity();
  const auto dw_02_in2 = knot2->acceleration();
  if (!T_10->active() || !w_01_in1->active() || !dw_01_in1->active()
    || !T_20->active() || !w_02_in2->active() || !dw_02_in2->active())
    throw std::runtime_error("extrapolation from a locked knot not implemented.");

  const auto T_10_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_10);
  const auto w_01_in1_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_01_in1);
  const auto dw_01_in1_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_01_in1);
  const auto T_20_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_20);
  const auto w_02_in2_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_02_in2);
  const auto dw_02_in2_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(dw_02_in2);
  if (!T_10_var || !w_01_in1_var || !dw_01_in1_var || !T_20_var || !w_02_in2_var || !dw_02_in2_var)
    throw std::runtime_error("trajectory states are not variables.");

  // Construct a knot for the interpolated state
  const auto T_q0_eval = getPoseInterpolator_(time, knot1, knot2);
  const auto w_0q_inq_eval = getVelocityInterpolator_(time, knot1, knot2);
  const auto dw_0q_inq_eval = getAccelerationInterpolator_(time, knot1, knot2);
  const auto knotq = Variable::MakeShared(time, T_q0_eval, w_0q_inq_eval, dw_0q_inq_eval);

  // Compute Jacobians
  // Note: jacKnot1 will return the negative of F as defined in
  // the state estimation textbook where we take the interpolation equations.
  // This doesn't apply to jacKnot2.
  const Eigen::Matrix<double, 18, 18> F_t1 = -getJacKnot1_(knot1, knotq);
  const Eigen::Matrix<double, 18, 18> E_t1 = getJacKnot2_(knot1, knotq);
  const Eigen::Matrix<double, 18, 18> F_2t = -getJacKnot1_(knotq, knot2);
  const Eigen::Matrix<double, 18, 18> E_2t = getJacKnot2_(knotq, knot2);

  // Prior inverse covariances
  const Eigen::Matrix<double, 18, 18> Qt1_inv = getQinv_((knotq->time() - knot1->time()).seconds(), Qc_diag_);
  const Eigen::Matrix<double, 18, 18> Q2t_inv = getQinv_((knot2->time() - knotq->time()).seconds(), Qc_diag_);

  // Covariance of knot1 and knot2
  const std::vector<StateVarBase::ConstPtr> state_var{T_10_var, w_01_in1_var, dw_01_in1_var, T_20_var, w_02_in2_var, dw_02_in2_var};
  const Eigen::Matrix<double, 36, 36> P_1n2 = cov.query(state_var);

  // Helper matrices
  Eigen::Matrix<double, 36, 18> A = Eigen::Matrix<double, 36, 18>::Zero();
  A.block<18, 18>(0, 0) = F_t1.transpose() * Qt1_inv * E_t1;
  A.block<18, 18>(18, 0) = E_2t.transpose() * Q2t_inv * F_2t;

  Eigen::Matrix<double, 36, 36> B = Eigen::Matrix<double, 36, 36>::Zero();
  B.block<18, 18>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
  B.block<18, 18>(18, 18) = E_2t.transpose() * Q2t_inv * E_2t;

  const Eigen::Matrix<double, 18, 18> F_21 = -getJacKnot1_(knot1, knot2);
  const Eigen::Matrix<double, 18, 18> E_21 = getJacKnot2_(knot1, knot2);
  const Eigen::Matrix<double, 18, 18> Q21_inv = getQinv_((knot2->time() - knot1->time()).seconds(), Qc_diag_);

  Eigen::Matrix<double, 36, 36> Pinv_comp = Eigen::Matrix<double, 36, 36>::Zero();
  Pinv_comp.block<18, 18>(0, 0) = F_21.transpose() * Q21_inv * F_21;
  Pinv_comp.block<18, 18>(18, 0) = -E_21.transpose() * Q21_inv * F_21;
  Pinv_comp.block<18, 18>(0, 18) = Pinv_comp.block<18, 18>(18, 0).transpose();
  Pinv_comp.block<18, 18>(18, 18) = E_21.transpose() * Q21_inv * E_21;

  // interpolated covariance
  const Eigen::Matrix<double, 18, 18> P_t_inv = E_t1.transpose() * Qt1_inv * E_t1 + F_2t.transpose() * Q2t_inv * F_2t -
                 A.transpose() * (P_1n2.inverse() + B - Pinv_comp).inverse() * A;

  return P_t_inv.inverse();
  // clang-format on
}

void Interface::addPosePrior(const Time& time, const PoseType& T_k0,
                             const Eigen::Matrix<double, 6, 6>& cov) {
  if (pose_prior_factor_ != nullptr)
    throw std::runtime_error("can only add one pose prior.");

  // Check that map is not empty
  if (knot_map_.empty()) throw std::runtime_error("knot map is empty.");

  // Try to find knot at same time
  auto it = knot_map_.find(time);
  if (it == knot_map_.end())
    throw std::runtime_error("no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the pose is not locked
  if (!knot->pose()->active())
    throw std::runtime_error("tried to add prior to locked pose.");

  // Set up loss function, noise model, and error function
  auto error_func = se3::se3_error(knot->pose(), T_k0);
  auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
  auto loss_func = L2LossFunc::MakeShared();

  // Create cost term
  pose_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
      error_func, noise_model, loss_func);
}

void Interface::addVelocityPrior(const Time& time, const VelocityType& w_0k_ink,
                                 const Eigen::Matrix<double, 6, 6>& cov) {
  if (vel_prior_factor_ != nullptr)
    throw std::runtime_error("can only add one velocity prior.");

  // Check that map is not empty
  if (knot_map_.empty()) throw std::runtime_error("knot map is empty.");

  // Try to find knot at same time
  auto it = knot_map_.find(time);
  if (it == knot_map_.end())
    throw std::runtime_error("no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the velocity is not locked
  if (!knot->velocity()->active())
    throw std::runtime_error("tried to add prior to locked velocity.");

  // Set up loss function, noise model, and error function
  auto error_func = vspace::vspace_error<6>(knot->velocity(), w_0k_ink);
  auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
  auto loss_func = L2LossFunc::MakeShared();

  // Create cost term
  vel_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
      error_func, noise_model, loss_func);
}

void Interface::addAccelerationPrior(const Time& time,
                                     const AccelerationType& dw_0k_ink,
                                     const Eigen::Matrix<double, 6, 6>& cov) {
  if (acc_prior_factor_ != nullptr)
    throw std::runtime_error("can only add one acceleration prior.");

  // Check that map is not empty
  if (knot_map_.empty()) throw std::runtime_error("knot map is empty.");

  // Try to find knot at same time
  auto it = knot_map_.find(time);
  if (it == knot_map_.end())
    throw std::runtime_error("no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the acceleration is not locked
  if (!knot->acceleration()->active())
    throw std::runtime_error("tried to add prior to locked acceleration.");

  // Set up loss function, noise model, and error function
  auto error_func = vspace::vspace_error<6>(knot->acceleration(), dw_0k_ink);
  auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
  auto loss_func = L2LossFunc::MakeShared();

  // Create cost term
  acc_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
      error_func, noise_model, loss_func);
}

void Interface::addStatePrior(const Time& time, const PoseType& T_k0,
                              const VelocityType& w_0k_ink,
                              const AccelerationType& dw_0k_ink,
                              const CovType& cov) {
  // Only allow adding 1 prior
  if ((pose_prior_factor_ != nullptr) || (vel_prior_factor_ != nullptr) ||
      (acc_prior_factor_ != nullptr))
    throw std::runtime_error("a pose/velocity prior already exists.");

  if (state_prior_factor_ != nullptr)
    throw std::runtime_error("can only add one state prior.");

  // Check that map is not empty
  if (knot_map_.empty()) throw std::runtime_error("knot map is empty.");

  // Try to find knot at unprovided time
  auto it = knot_map_.find(time);
  if (it == knot_map_.end())
    throw std::runtime_error("no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the pose is not locked
  if ((!knot->pose()->active()) || (!knot->velocity()->active()) ||
      (!knot->acceleration()->active()))
    throw std::runtime_error("tried to add prior to locked state.");

  auto pose_error = se3::se3_error(knot->pose(), T_k0);
  auto velo_error = vspace::vspace_error<6>(knot->velocity(), w_0k_ink);
  auto acc_error = vspace::vspace_error<6>(knot->acceleration(), dw_0k_ink);
  auto error_temp = vspace::merge<6, 6>(pose_error, velo_error);
  auto error_func = vspace::merge<12, 6>(error_temp, acc_error);
  auto noise_model = StaticNoiseModel<18>::MakeShared(cov);
  auto loss_func = L2LossFunc::MakeShared();

  // Create cost term
  state_prior_factor_ = WeightedLeastSqCostTerm<18>::MakeShared(
      error_func, noise_model, loss_func);
}

void Interface::addPriorCostTerms(Problem& problem) const {
  // If empty, return none
  if (knot_map_.empty()) return;

  // Check for pose or velocity priors
  if (pose_prior_factor_ != nullptr) problem.addCostTerm(pose_prior_factor_);
  if (vel_prior_factor_ != nullptr) problem.addCostTerm(vel_prior_factor_);
  if (acc_prior_factor_ != nullptr) problem.addCostTerm(acc_prior_factor_);

  // All prior factors will use an L2 loss function
  const auto loss_function = std::make_shared<L2LossFunc>();

  // Initialize iterators
  auto it1 = knot_map_.begin();
  auto it2 = it1;
  ++it2;

  // Iterate through all states.. if any are unlocked, supply a prior term
  for (; it2 != knot_map_.end(); ++it1, ++it2) {
    // Get knots
    const auto& knot1 = it1->second;
    const auto& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if (knot1->pose()->active() || knot1->velocity()->active() ||
        knot1->acceleration()->active() || knot2->pose()->active() ||
        knot2->velocity()->active() || knot2->acceleration()->active()) {
      // Generate information matrix for GP prior factor
      auto Qinv = getQinv_((knot2->time() - knot1->time()).seconds(), Qc_diag_);
      const auto noise_model =
          std::make_shared<StaticNoiseModel<18>>(Qinv, NoiseType::INFORMATION);
      //
      const auto error_function = getPriorFactor_(knot1, knot2);
      // Create cost term
      const auto cost_term = std::make_shared<WeightedLeastSqCostTerm<18>>(
          error_function, noise_model, loss_function);
      //
      problem.addCostTerm(cost_term);
    }
  }
}

Eigen::Matrix<double, 18, 18> Interface::getJacKnot1_(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
  return getJacKnot1(knot1, knot2);
}

Eigen::Matrix<double, 18, 18> Interface::getJacKnot2_(
    const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
  return getJacKnot2(knot1, knot2);
}

Eigen::Matrix<double, 18, 18> Interface::getQ_(
    const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
  return getQ(dt, Qc_diag);
}

Eigen::Matrix<double, 18, 18> Interface::getQinv_(
    const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
  return getQinv(dt, Qc_diag);
}

auto Interface::getPoseInterpolator_(const Time& time,
                                     const Variable::ConstPtr& knot1,
                                     const Variable::ConstPtr& knot2) const
    -> Evaluable<PoseType>::Ptr {
  return PoseInterpolator::MakeShared(time, knot1, knot2);
}

auto Interface::getVelocityInterpolator_(const Time& time,
                                         const Variable::ConstPtr& knot1,
                                         const Variable::ConstPtr& knot2) const
    -> Evaluable<VelocityType>::Ptr {
  return VelocityInterpolator::MakeShared(time, knot1, knot2);
}

auto Interface::getAccelerationInterpolator_(
    const Time& time, const Variable::ConstPtr& knot1,
    const Variable::ConstPtr& knot2) const -> Evaluable<AccelerationType>::Ptr {
  return AccelerationInterpolator::MakeShared(time, knot1, knot2);
}

auto Interface::getPoseExtrapolator_(const Time& time,
                                     const Variable::ConstPtr& knot) const
    -> Evaluable<PoseType>::Ptr {
  return PoseExtrapolator::MakeShared(time, knot);
}

auto Interface::getVelocityExtrapolator_(const Time& time,
                                         const Variable::ConstPtr& knot) const
    -> Evaluable<VelocityType>::Ptr {
  return VelocityExtrapolator::MakeShared(time, knot);
}

auto Interface::getAccelerationExtrapolator_(
    const Time& time, const Variable::ConstPtr& knot) const
    -> Evaluable<AccelerationType>::Ptr {
  return AccelerationExtrapolator::MakeShared(time, knot);
}

auto Interface::getPriorFactor_(const Variable::ConstPtr& knot1,
                                const Variable::ConstPtr& knot2) const
    -> Evaluable<Eigen::Matrix<double, 18, 1>>::Ptr {
  return PriorFactor::MakeShared(knot1, knot2);
}

}  // namespace const_acc
}  // namespace traj
}  // namespace steam