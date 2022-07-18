#include "steam/trajectory/const_vel/interface.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/noise_model/static_noise_model.hpp"
#include "steam/trajectory/const_vel/pose_extrapolator.hpp"
#include "steam/trajectory/const_vel/pose_interpolator.hpp"
#include "steam/trajectory/const_vel/prior_factor.hpp"
#include "steam/trajectory/const_vel/velocity_interpolator.hpp"

namespace steam {
namespace traj {
namespace const_vel {

auto Interface::MakeShared(const Eigen::Matrix<double, 6, 6>& Qc_inv) -> Ptr {
  return std::make_shared<Interface>(Qc_inv);
}

Interface::Interface(const Eigen::Matrix<double, 6, 6>& Qc_inv)
    : Qc_inv_(Qc_inv) {}

void Interface::add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                    const Evaluable<VelocityType>::Ptr& w_0k_ink) {
  if (knotMap_.find(time) != knotMap_.end())
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] adding knot at duplicated time.");
  const auto knot = std::make_shared<Variable>(time, T_k0, w_0k_ink);
  knotMap_.insert(knotMap_.end(), std::pair<Time, Variable::Ptr>(time, knot));
}

auto Interface::getPoseInterpolator(const Time& time) const
    -> Evaluable<PoseType>::ConstPtr {
  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error(
        "[ConstVelTraj][getPoseInterpolator] map was empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knotMap_.lower_bound(time);

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {
    --it1;  // should be safe, as we checked that the map was not empty..
    const auto& endKnot = it1->second;
    const auto T_t_k = PoseExtrapolator::MakeShared(time - endKnot->getTime(),
                                                    endKnot->getVelocity());
    return se3::compose(T_t_k, endKnot->getPose());
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
    // return state variable exactly (no interp)
    return it1->second->getPose();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {
    const auto& startKnot = it1->second;
    const auto T_t_k = PoseExtrapolator::MakeShared(time - startKnot->getTime(),
                                                    startKnot->getVelocity());
    return se3::compose(T_t_k, startKnot->getPose());
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error(
        "Requested trajectory evaluator at an invalid time. This exception "
        "should not trigger... report to a STEAM contributor.");
  }

  // Create interpolated evaluator
  return PoseInterpolator::MakeShared(time, it1->second, it2->second);
}

auto Interface::getVelocityInterpolator(const Time& time) const
    -> Evaluable<VelocityType>::ConstPtr {
  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[ConstVelTraj][getEvaluator] map was empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knotMap_.lower_bound(time);

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {
    --it1;  // should be safe, as we checked that the map was not empty..
    const auto& endKnot = it1->second;
    return endKnot->getVelocity();
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
    // return state variable exactly (no interp)
    return it1->second->getVelocity();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {
    const auto& startKnot = it1->second;
    return startKnot->getVelocity();
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error(
        "Requested trajectory evaluator at an invalid time. This exception "
        "should not trigger... report to a STEAM contributor.");
  }

  // Create interpolated evaluator
  return VelocityInterpolator::MakeShared(time, it1->second, it2->second);
}

auto Interface::getCovariance(const Covariance& cov, const Time& time)
    -> CovType {
  // clang-format off

  //
  if (knotMap_.empty()) throw std::runtime_error("[ConstVelTraj][getCovariance] map was empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knotMap_.lower_bound(time);

  // extrapolate after last entry
  if (it1 == knotMap_.end()) {
    --it1;  // should be safe, as we checked that the map was not empty..

    const auto& endKnot = it1->second;
    const auto T_k0 = endKnot->getPose();
    const auto w_0k_ink = endKnot->getVelocity();
    if (!T_k0->active() || !w_0k_ink->active())
      throw std::runtime_error("[ConstVelTraj][getCovariance] extrapolation from a locked knot not implemented.");

    const auto T_k0_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_k0);
    const auto w_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_0k_ink);
    if (!T_k0_var || !w_0k_ink_var)
      throw std::runtime_error("[ConstVelTraj][getCovariance] trajectory states are not variables.");

    // Construct a knot for the extrapolated state
    const auto T_t_k = PoseExtrapolator::MakeShared(time - endKnot->getTime(), endKnot->getVelocity());
    const auto T_t_0 = se3::compose(T_t_k, endKnot->getPose());  // mean of extrapolation
    const auto extrap_knot = Variable::MakeShared(time, T_t_0, endKnot->getVelocity());

    // Compute Jacobians
    // Note: jacKnot1 will return the negative of F as defined in
    // the state estimation textbook where we take the interpolation equations.
    // This doesn't apply to jacKnot2.
    auto F_t1 = -PriorFactor::jacKnot1(endKnot, extrap_knot);
    auto E_t1_inv = PriorFactor::jacKnot2(endKnot, extrap_knot).inverse();

    // Prior covariance
    Eigen::Matrix<double, 12, 12> Qt1_inv = computeQinv((extrap_knot->getTime() - endKnot->getTime()).seconds());

    // end knot covariance
    std::vector<StateVarBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var};
    Eigen::Matrix<double, 12, 12> P_end = cov.query(state_var);

    // Compute covariance
    return E_t1_inv * (F_t1 * P_end * F_t1.transpose() + Qt1_inv.inverse()) * E_t1_inv.transpose();
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
    const auto& knot = it1->second;
    const auto T_k0 = knot->getPose();
    const auto w_0k_ink = knot->getVelocity();
    if (!T_k0->active() || !w_0k_ink->active())
      throw std::runtime_error("[ConstVelTraj][getCovariance] extrapolation from a locked knot not implemented.");

    const auto T_k0_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_k0);
    const auto w_0k_ink_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_0k_ink);
    if (!T_k0_var || !w_0k_ink_var)
      throw std::runtime_error("[ConstVelTraj][getCovariance] trajectory states are not variables.");

    std::vector<StateVarBase::ConstPtr> state_var{T_k0_var, w_0k_ink_var};
    return cov.query(state_var);
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {
    throw std::runtime_error("[ConstVelTraj][getCovariance] Requested covariance before first time.");
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;

  const auto& knot1 = it1->second;
  const auto T_10 = knot1->getPose();
  const auto w_01_in1 = knot1->getVelocity();
  const auto& knot2 = it2->second;
  const auto T_20 = knot2->getPose();
  const auto w_02_in2 = knot2->getVelocity();
  if (!T_10->active() || !w_01_in1->active() || !T_20->active() || !w_02_in2->active())
    throw std::runtime_error("[ConstVelTraj][getCovariance] extrapolation from a locked knot not implemented.");

  const auto T_10_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_10);
  const auto w_01_in1_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_01_in1);
  const auto T_20_var = std::dynamic_pointer_cast<se3::SE3StateVar>(T_20);
  const auto w_02_in2_var = std::dynamic_pointer_cast<vspace::VSpaceStateVar<6>>(w_02_in2);
  if (!T_10_var || !w_01_in1_var || !T_20_var || !w_02_in2_var)
    throw std::runtime_error("[ConstVelTraj][getCovariance] trajectory states are not variables.");

  // Construct a knot for the interpolated state
  auto T_q0_eval = PoseInterpolator::MakeShared(time, knot1, knot2);
  auto w_0q_inq_eval = VelocityInterpolator::MakeShared(time, knot1, knot2);
  auto knotq = Variable::MakeShared(time, T_q0_eval, w_0q_inq_eval);

  // Compute Jacobians
  // Note: jacKnot1 will return the negative of F as defined in
  // the state estimation textbook where we take the interpolation equations.
  // This doesn't apply to jacKnot2.
  auto F_t1 = -PriorFactor::jacKnot1(knot1, knotq);
  auto E_t1 = PriorFactor::jacKnot2(knot1, knotq);
  auto F_2t = -PriorFactor::jacKnot1(knotq, knot2);
  auto E_2t = PriorFactor::jacKnot2(knotq, knot2);

  // Prior inverse covariances
  Eigen::Matrix<double, 12, 12> Qt1_inv = computeQinv((knotq->getTime() - knot1->getTime()).seconds());
  Eigen::Matrix<double, 12, 12> Q2t_inv = computeQinv((knot2->getTime() - knotq->getTime()).seconds());

  // Covariance of knot1 and knot2
  std::vector<StateVarBase::ConstPtr> state_var{T_10_var, w_01_in1_var, T_20_var, w_02_in2_var};
  Eigen::Matrix<double, 24, 24> P_1n2 = cov.query(state_var);

  // Helper matrices
  Eigen::Matrix<double, 24, 12> A = Eigen::Matrix<double, 24, 12>::Zero();
  A.block<12, 12>(0, 0) = F_t1.transpose() * Qt1_inv * E_t1;
  A.block<12, 12>(12, 0) = E_2t.transpose() * Q2t_inv * F_2t;

  Eigen::Matrix<double, 24, 24> B = Eigen::Matrix<double, 24, 24>::Zero();
  B.block<12, 12>(0, 0) = F_t1.transpose() * Qt1_inv * F_t1;
  B.block<12, 12>(12, 12) = E_2t.transpose() * Q2t_inv * E_2t;

  auto F_21 = -PriorFactor::jacKnot1(knot1, knot2);
  auto E_21 = PriorFactor::jacKnot2(knot1, knot2);
  Eigen::Matrix<double, 12, 12> Q21_inv = computeQinv((knot2->getTime() - knot1->getTime()).seconds());
  Eigen::Matrix<double, 24, 24> Pinv_comp = Eigen::Matrix<double, 24, 24>::Zero();
  Pinv_comp.block<12, 12>(0, 0) = F_21.transpose() * Q21_inv * F_21;
  Pinv_comp.block<12, 12>(12, 0) = -E_21.transpose() * Q21_inv * F_21;
  Pinv_comp.block<12, 12>(0, 12) = Pinv_comp.block<12, 12>(12, 0).transpose();
  Pinv_comp.block<12, 12>(12, 12) = E_21.transpose() * Q21_inv * E_21;

  // interpolated covariance
  auto P_t_inv = E_t1.transpose() * Qt1_inv * E_t1 +
                 F_2t.transpose() * Q2t_inv * F_2t -
                 A.transpose() * (P_1n2.inverse() + B - Pinv_comp).inverse() * A;

  return P_t_inv.inverse();

  // clang-format on
}

void Interface::addPosePrior(const Time& time, const PoseType& T_k0,
                             const Eigen::Matrix<double, 6, 6>& cov) {
  if (state_prior_factor_ != nullptr)
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] a state prior already exists.");

  if (pose_prior_factor_ != nullptr)
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] can only add one pose prior.");

  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[ConstVelTraj][addPosePrior] map was empty.");

  // Try to find knot at same time
  auto it = knotMap_.find(time);
  if (it == knotMap_.end())
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the pose is not locked
  if (!knot->getPose()->active())
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] tried to add prior to locked pose.");

  // Set up loss function, noise model, and error function
  auto error_func = se3::se3_error(knot->getPose(), T_k0);
  auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
  auto loss_func = L2LossFunc::MakeShared();

  // Create cost term
  pose_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
      error_func, noise_model, loss_func);
}

void Interface::addVelocityPrior(const Time& time, const VelocityType& w_0k_ink,
                                 const Eigen::Matrix<double, 6, 6>& cov) {
  // Only allow adding 1 prior
  if (state_prior_factor_ != nullptr)
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] a state prior already exists.");

  if (vel_prior_factor_ != nullptr)
    throw std::runtime_error(
        "[ConstVelTraj][addVelocityPrior] can only add one velocity prior.");

  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[ConstVelTraj][addVelocityPrior] map was empty.");

  // Try to find knot at same time
  auto it = knotMap_.find(time);
  if (it == knotMap_.end())
    throw std::runtime_error(
        "[ConstVelTraj][addVelocityPrior] no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the velocity is not locked
  if (!knot->getVelocity()->active())
    throw std::runtime_error(
        "[ConstVelTraj][addVelocityPrior] tried to add prior to locked "
        "velocity.");

  // Set up loss function, noise model, and error function
  auto error_func = vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink);
  auto noise_model = StaticNoiseModel<6>::MakeShared(cov);
  auto loss_func = L2LossFunc::MakeShared();

  // Create cost term
  vel_prior_factor_ = WeightedLeastSqCostTerm<6>::MakeShared(
      error_func, noise_model, loss_func);
}

void Interface::addStatePrior(const Time& time, const PoseType& T_k0,
                              const VelocityType& w_0k_ink,
                              const Eigen::Matrix<double, 12, 12>& cov) {
  // Only allow adding 1 prior
  if ((pose_prior_factor_ != nullptr) || (vel_prior_factor_ != nullptr))
    throw std::runtime_error(
        "[ConstVelTraj][addVelocityPrior] a pose/velocity prior already "
        "exists.");

  if (state_prior_factor_ != nullptr)
    throw std::runtime_error(
        "[ConstVelTraj][addVelocityPrior] can only add one state prior.");

  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[ConstVelTraj][addPosePrior] map was empty.");

  // Try to find knot at unprovided time
  auto it = knotMap_.find(time);
  if (it == knotMap_.end())
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the pose is not locked
  if ((!knot->getPose()->active()) || (!knot->getVelocity()->active()))
    throw std::runtime_error(
        "[ConstVelTraj][addPosePrior] tried to add prior to locked state.");

  auto pose_error = se3::se3_error(knot->getPose(), T_k0);
  auto velo_error = vspace::vspace_error<6>(knot->getVelocity(), w_0k_ink);
  auto error_func = vspace::merge<6, 6>(pose_error, velo_error);
  auto noise_model = StaticNoiseModel<12>::MakeShared(cov);
  auto loss_func = L2LossFunc::MakeShared();

  // Create cost term
  state_prior_factor_ = WeightedLeastSqCostTerm<12>::MakeShared(
      error_func, noise_model, loss_func);
}

void Interface::addPriorCostTerms(OptimizationProblem& problem) const {
  // If empty, return none
  if (knotMap_.empty()) return;

  // Check for pose or velocity priors
  if (pose_prior_factor_ != nullptr) problem.addCostTerm(pose_prior_factor_);
  if (vel_prior_factor_ != nullptr) problem.addCostTerm(vel_prior_factor_);
  if (state_prior_factor_ != nullptr) problem.addCostTerm(state_prior_factor_);

  // All prior factors will use an L2 loss function
  const auto loss_function = std::make_shared<L2LossFunc>();

  // Initialize iterators
  auto it1 = knotMap_.begin();
  auto it2 = it1;
  ++it2;

  // Iterate through all states.. if any are unlocked, supply a prior term
  for (; it2 != knotMap_.end(); ++it1, ++it2) {
    // Get knots
    const auto& knot1 = it1->second;
    const auto& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if (knot1->getPose()->active() || knot1->getVelocity()->active() ||
        knot2->getPose()->active() || knot2->getVelocity()->active()) {
      // Generate 12 x 12 information matrix for GP prior factor
      Eigen::Matrix<double, 12, 12> Qi_inv =
          computeQinv((knot2->getTime() - knot1->getTime()).seconds());
      const auto noise_model = std::make_shared<StaticNoiseModel<12>>(
          Qi_inv, NoiseType::INFORMATION);
      //
      const auto error_function = PriorFactor::MakeShared(knot1, knot2);
      // Create cost term
      const auto cost_term = std::make_shared<WeightedLeastSqCostTerm<12>>(
          error_function, noise_model, loss_function);
      //
      problem.addCostTerm(cost_term);
    }
  }
}

auto Interface::computeQinv(const double& deltatime) const
    -> Eigen::Matrix<double, 12, 12> {
  Eigen::Matrix<double, 12, 12> Qi_inv;
  double one_over_dt = 1.0 / deltatime;
  double one_over_dt2 = one_over_dt * one_over_dt;
  double one_over_dt3 = one_over_dt2 * one_over_dt;
  Qi_inv.block<6, 6>(0, 0) = 12.0 * one_over_dt3 * Qc_inv_;
  Qi_inv.block<6, 6>(6, 0) = Qi_inv.block<6, 6>(0, 6) =
      -6.0 * one_over_dt2 * Qc_inv_;
  Qi_inv.block<6, 6>(6, 6) = 4.0 * one_over_dt * Qc_inv_;
  return Qi_inv;
}

}  // namespace const_vel
}  // namespace traj
}  // namespace steam