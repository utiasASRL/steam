#include "steam/trajectory/traj_interface.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/problem/LossFunctions.hpp"
#include "steam/problem/NoiseModel.hpp"
#include "steam/trajectory/traj_pose_extrapolator.hpp"
#include "steam/trajectory/traj_pose_interpolator.hpp"
#include "steam/trajectory/traj_prior_factor.hpp"
#include "steam/trajectory/traj_velocity_interpolator.hpp"

namespace steam {
namespace traj {

TrajInterface::TrajInterface(const bool allowExtrapolation)
    : Qc_inv_(Eigen::Matrix<double, 6, 6>::Identity()),
      allowExtrapolation_(allowExtrapolation) {}

TrajInterface::TrajInterface(const Eigen::Matrix<double, 6, 6>& Qc_inv,
                             const bool allowExtrapolation)
    : Qc_inv_(Qc_inv), allowExtrapolation_(allowExtrapolation) {}

void TrajInterface::add(const TrajVar::Ptr& knot) {
  knotMap_.insert(knotMap_.end(),
                  std::pair<Time, TrajVar::Ptr>(knot->getTime(), knot));
}

void TrajInterface::add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                        const Evaluable<VelocityType>::Ptr& w_0k_ink) {
  add(std::make_shared<TrajVar>(time, T_k0, w_0k_ink));
}

void TrajInterface::add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                        const Evaluable<VelocityType>::Ptr& w_0k_ink,
                        const CovType& cov) {
  add(std::make_shared<TrajVar>(time, T_k0, w_0k_ink, cov));
}

auto TrajInterface::getPoseInterpolator(const Time& time) const
    -> Evaluable<PoseType>::ConstPtr {
  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[STEAMTraj][getPoseInterpolator] map was empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knotMap_.lower_bound(time);

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {
    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      --it1;  // should be safe, as we checked that the map was not empty..
      const auto& endKnot = it1->second;
      const auto T_t_k = TrajPoseExtrapolator::MakeShared(
          time - endKnot->getTime(), endKnot->getVelocity());
      return se3::compose(T_t_k, endKnot->getPose());
    } else {
      throw std::runtime_error(
          "Requested trajectory evaluator at an invalid time.");
    }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
    // return state variable exactly (no interp)
    return it1->second->getPose();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {
    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      const auto& startKnot = it1->second;
      const auto T_t_k = TrajPoseExtrapolator::MakeShared(
          time - startKnot->getTime(), startKnot->getVelocity());
      return se3::compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error(
          "Requested trajectory evaluator at an invalid time.");
    }
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
  return TrajPoseInterpolator::MakeShared(time, it1->second, it2->second);
}

auto TrajInterface::getVelocityInterpolator(const Time& time) const
    -> Evaluable<VelocityType>::ConstPtr {
  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[STEAMTraj][getEvaluator] map was empty");

  // Get iterator to first element with time equal to or greater than 'time'
  auto it1 = knotMap_.lower_bound(time);

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {
    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      --it1;  // should be safe, as we checked that the map was not empty..
      const auto& endKnot = it1->second;
      return endKnot->getVelocity();
    } else {
      throw std::runtime_error(
          "Requested trajectory evaluator at an invalid time.");
    }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
    // return state variable exactly (no interp)
    return it1->second->getVelocity();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {
    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      const auto& startKnot = it1->second;
      return startKnot->getVelocity();
    } else {
      throw std::runtime_error(
          "Requested trajectory evaluator at an invalid time.");
    }
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
  return TrajVelocityInterpolator::MakeShared(time, it1->second, it2->second);
}

void TrajInterface::addPosePrior(const Time& time, const PoseType& T_k0,
                                 const Eigen::Matrix<double, 6, 6>& cov) {
  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[STEAMTraj][addPosePrior] map was empty.");

  // Try to find knot at same time
  auto it = knotMap_.find(time);
  if (it == knotMap_.end())
    throw std::runtime_error(
        "[STEAMTraj][addPosePrior] no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the pose is not locked
  if (!knot->getPose()->active())
    throw std::runtime_error(
        "[STEAMTraj][addPosePrior] tried to add prior to locked pose.");

  // Set up loss function, noise model, and error function
  const auto T_k0_meas = se3::SE3StateVar::MakeShared(T_k0);
  T_k0_meas->locked() = true;
  const auto error_function =
      se3::tran2vec(se3::compose(T_k0_meas, se3::inverse(knot->getPose())));
  const auto noise_model = std::make_shared<StaticNoiseModel<6>>(cov);
  const auto loss_function = std::make_shared<L2LossFunc>();

  // Create cost term
  posePriorFactor_ = std::make_shared<WeightedLeastSqCostTerm<6>>(
      error_function, noise_model, loss_function);
}

void TrajInterface::addVelocityPrior(const Time& time,
                                     const VelocityType& w_0k_ink,
                                     const Eigen::Matrix<double, 6, 6>& cov) {
  // Check that map is not empty
  if (knotMap_.empty())
    throw std::runtime_error("[STEAMTraj][addVelocityPrior] map was empty.");

  // Try to find knot at same time
  auto it = knotMap_.find(time);
  if (it == knotMap_.end())
    throw std::runtime_error(
        "[STEAMTraj][addVelocityPrior] no knot at provided time.");

  // Get reference
  const auto& knot = it->second;

  // Check that the velocity is not locked
  if (!knot->getVelocity()->active())
    throw std::runtime_error(
        "[STEAMTraj][addVelocityPrior] tried to add prior to locked pose.");

  // Set up loss function, noise model, and error function
  const auto w_0k_ink_meas = vspace::VSpaceStateVar<6>::MakeShared(w_0k_ink);
  w_0k_ink_meas->locked() = true;
  const auto error_function =
      vspace::add<6>(w_0k_ink_meas, vspace::neg<6>(knot->getVelocity()));
  const auto noise_model = std::make_shared<StaticNoiseModel<6>>(cov);
  const auto loss_function = std::make_shared<L2LossFunc>();

  // Create cost term
  velocityPriorFactor_ = std::make_shared<WeightedLeastSqCostTerm<6>>(
      error_function, noise_model, loss_function);
}

void TrajInterface::addPriorCostTerms(OptimizationProblem& problem) const {
  // If empty, return none
  if (knotMap_.empty()) return;

  // Check for pose or velocity priors
  if (posePriorFactor_) problem.addCostTerm(posePriorFactor_);
  if (velocityPriorFactor_) problem.addCostTerm(velocityPriorFactor_);

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
      Eigen::Matrix<double, 12, 12> Qi_inv;
      double one_over_dt =
          1.0 / (knot2->getTime() - knot1->getTime()).seconds();
      double one_over_dt2 = one_over_dt * one_over_dt;
      double one_over_dt3 = one_over_dt2 * one_over_dt;
      Qi_inv.block<6, 6>(0, 0) = 12.0 * one_over_dt3 * Qc_inv_;
      Qi_inv.block<6, 6>(6, 0) = Qi_inv.block<6, 6>(0, 6) =
          -6.0 * one_over_dt2 * Qc_inv_;
      Qi_inv.block<6, 6>(6, 6) = 4.0 * one_over_dt * Qc_inv_;
      const auto noise_model =
          std::make_shared<StaticNoiseModel<12>>(Qi_inv, steam::INFORMATION);
      //
      const auto error_function = TrajPriorFactor::MakeShared(knot1, knot2);
      // Create cost term
      const auto cost_term = std::make_shared<WeightedLeastSqCostTerm<12>>(
          error_function, noise_model, loss_function);
      //
      problem.addCostTerm(cost_term);
    }
  }
}

}  // namespace traj
}  // namespace steam
