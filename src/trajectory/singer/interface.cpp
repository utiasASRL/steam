#include "steam/trajectory/singer/interface.hpp"

#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/noise_model/static_noise_model.hpp"
#include "steam/trajectory/singer/helper.hpp"
#include "steam/trajectory/singer/pose_interpolator.hpp"
#include "steam/trajectory/singer/prior_factor.hpp"
#include "steam/trajectory/singer/velocity_interpolator.hpp"

namespace steam {
namespace traj {
namespace singer {

auto Interface::MakeShared(const Eigen::Matrix<double, 6, 1>& alpha_diag,
                           const Eigen::Matrix<double, 6, 1>& Qc_diag) -> Ptr {
  return std::make_shared<Interface>(alpha_diag, Qc_diag);
}

Interface::Interface(const Eigen::Matrix<double, 6, 1>& alpha_diag,
                     const Eigen::Matrix<double, 6, 1>& Qc_diag)
    : alpha_diag_(alpha_diag), Qc_diag_(Qc_diag) {}

void Interface::add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                    const Evaluable<VelocityType>::Ptr& w_0k_ink,
                    const Evaluable<VelocityType>::Ptr& dw_0k_ink) {
  if (knot_map_.find(time) != knot_map_.end())
    throw std::runtime_error("adding knot at duplicated time");
  const auto knot = std::make_shared<Variable>(time, T_k0, w_0k_ink, dw_0k_ink);
  knot_map_.insert(knot_map_.end(), std::pair<Time, Variable::Ptr>(time, knot));
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
    throw std::runtime_error("pose extrapolation not implemented");
    // const auto& endKnot = it1->second;
    // const auto T_t_k = PoseExtrapolator::MakeShared(time - endKnot->time(),
    //                                                 endKnot->velocity());
    // return se3::compose(T_t_k, endKnot->pose());
  }

  // Check if we requested time exactly
  if (it1->second->time() == time) return it1->second->pose();

  // Check if we requested before first time
  if (it1 == knot_map_.begin()) {
    throw std::runtime_error("pose extrapolation not implemented");
    // const auto& startKnot = it1->second;
    // const auto T_t_k = PoseExtrapolator::MakeShared(time - startKnot->time(),
    //                                                 startKnot->velocity());
    // return se3::compose(T_t_k, startKnot->pose());
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;
  if (time <= it1->second->time() || time >= it2->second->time())
    throw std::runtime_error("Requested interpolation at an invalid time");

  // Create interpolated evaluator
  return PoseInterpolator::MakeShared(time, it1->second, it2->second,
                                      alpha_diag_);
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
    throw std::runtime_error("velocity extrapolation not implemented");
    // const auto& endKnot = it1->second;
    // return endKnot->velocity();
  }

  // Check if we requested time exactly
  if (it1->second->time() == time) return it1->second->velocity();

  // Check if we requested before first time
  if (it1 == knot_map_.begin()) {
    throw std::runtime_error("pose extrapolation not implemented");
    // const auto& startKnot = it1->second;
    // return startKnot->velocity();
  }

  // Get iterators bounding the time interval
  auto it2 = it1;
  --it1;
  if (time <= it1->second->time() || time >= it2->second->time())
    throw std::runtime_error("Requested interpolation at an invalid time");

  // Create interpolated evaluator
  return VelocityInterpolator::MakeShared(time, it1->second, it2->second,
                                          alpha_diag_);
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
      auto Qinv = getQ((knot2->time() - knot1->time()).seconds(), Qc_diag_);
      const auto noise_model =
          std::make_shared<StaticNoiseModel<18>>(Qinv, NoiseType::COVARIANCE);
      //
      const auto error_function =
          PriorFactor::MakeShared(knot1, knot2, alpha_diag_);
      // Create cost term
      const auto cost_term = std::make_shared<WeightedLeastSqCostTerm<18>>(
          error_function, noise_model, loss_function);
      //
      problem.addCostTerm(cost_term);
    }
  }
}

}  // namespace singer
}  // namespace traj
}  // namespace steam