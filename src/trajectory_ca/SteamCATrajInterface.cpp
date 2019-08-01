//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamCATrajInterface.cpp
///
/// \author Tim Tang, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory_ca/SteamCATrajInterface.hpp>

#include <lgmath.hpp>

#include <steam/trajectory_ca/SteamCATrajPoseInterpEval.hpp>
#include <steam/trajectory_ca/SteamCATrajPriorFactor.hpp>
#include <steam/evaluator/samples/VectorSpaceErrorEval.hpp>

#include <steam/evaluator/blockauto/transform/TransformEvalOperations.hpp>
#include <steam/evaluator/blockauto/transform/ConstVelTransformEvaluator.hpp>
#include <steam/evaluator/blockauto/transform/ConstAccTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
///        Note, without providing Qc, the trajectory can be used safely for interpolation,
///        but should not be used for estimation.
//////////////////////////////////////////////////////////////////////////////////////////////
SteamCATrajInterface::SteamCATrajInterface(bool allowExtrapolation) :
  Qc_inv_(Eigen::Matrix<double,6,6>::Identity()), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamCATrajInterface::SteamCATrajInterface(const Eigen::Matrix<double,6,6>& Qc_inv,
                                       bool allowExtrapolation) :
  Qc_inv_(Qc_inv), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::add(const SteamCATrajVar::Ptr& knot) {

  // Todo, check that time does not already exist in map?

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamCATrajVar::Ptr>(knot->getTime().nanosecs(), knot));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::add(const steam::Time& time,
                             const se3::TransformEvaluator::Ptr& T_k0,
                             const VectorSpaceStateVar::Ptr& velocity,
                             const VectorSpaceStateVar::Ptr& acceleration) {

  // Check velocity input
  if (velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }

  // Check acceleration input
  if (acceleration->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid acceleration size");
  }

  // Todo, check that time does not already exist in map?

  // Make knot
  SteamCATrajVar::Ptr newEntry(new SteamCATrajVar(time, T_k0, velocity, acceleration));

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamCATrajVar::Ptr>(time.nanosecs(), newEntry));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
TransformEvaluator::ConstPtr SteamCATrajInterface::getInterpPoseEval(const steam::Time& time) const {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

    // If we allow extrapolation, return constant-acceleration interpolated entry
    if (allowExtrapolation_) {
      --it1; // should be safe, as we checked that the map was not empty..
      const SteamCATrajVar::Ptr& endKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstAccTransformEvaluator::MakeShared(endKnot->getVelocity(),
                                                 endKnot->getAcceleration(),
                                                 time - endKnot->getTime());
      return compose(T_t_k, endKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {

    // return state variable exactly (no interp)
    return it1->second->getPose();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {

    // If we allow extrapolation, return constant-acceleration interpolated entry
    if (allowExtrapolation_) {
      const SteamCATrajVar::Ptr& startKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
      ConstAccTransformEvaluator::MakeShared(startKnot->getVelocity(),
                                             startKnot->getAcceleration(),
                                             time - startKnot->getTime());
      return compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // Create interpolated evaluator
  return SteamCATrajPoseInterpEval::MakeShared(time, it1->second, it2->second);
}

Eigen::VectorXd SteamCATrajInterface::getVelocity(const steam::Time& time) {
  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or greater than 'time'
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it1
     = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

   // If we allow extrapolation, return constant-velocity interpolated entry
   if (allowExtrapolation_) {
     --it1; // should be safe, as we checked that the map was not empty..
     const SteamCATrajVar::Ptr& endKnot = it1->second;
     return endKnot->getVelocity()->getValue();
   } else {
     throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
   }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
     const SteamCATrajVar::Ptr& knot = it1->second;
     // return state variable exactly (no interp)
     return knot->getVelocity()->getValue();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {
    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
     const SteamCATrajVar::Ptr& startKnot = it1->second;
     return startKnot->getVelocity()->getValue();
    } else {
     throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                            "should not trigger... report to a STEAM contributor.");
  }

  // OK, we actually need to interpolate.
  // Follow a similar setup to SteamCATrajPoseInterpEval

  // Convenience defs
  auto &knot1 = it1->second;
  auto &knot2 = it2->second;

  double t1 = knot1->getTime().seconds();
  double t2 = knot2->getTime().seconds();
  double tau = time.seconds();

  // Cheat by calculating deltas wrt t1, so we can avoid super large values
  tau = tau-t1;
  t2 = t2-t1;
  t1 = 0;
  
  double T = (knot2->getTime() - knot1->getTime()).seconds();
  double delta_tau = (time - knot1->getTime()).seconds();
  double delta_kappa = (knot2->getTime()-time).seconds();

  // std::cout << t1 << " " << t2 << " " << tau << std::endl;

  double T2 = T*T;
  double T3 = T2*T;
  double T4 = T3*T;
  double T5 = T4*T;

  double delta_tau2 = delta_tau*delta_tau;
  double delta_tau3 = delta_tau2*delta_tau;
  double delta_tau4 = delta_tau3*delta_tau;
  double delta_kappa2 = delta_kappa*delta_kappa;
  double delta_kappa3 = delta_kappa2*delta_kappa;

  // Calculate 'omega' interpolation values
  double omega11 = delta_tau3/T5*(t1*t1 - 5*t1*t2 + 3*t1*tau + 10*t2*t2 - 15*t2*tau + 6*tau*tau);
  double omega12 = delta_tau3*delta_kappa/T4*(t1 - 4*t2 + 3*tau);
  double omega13 = delta_tau3*delta_kappa2/(2*T3);

  double omega21 = 30*delta_tau2/T5*(6*delta_kappa2-4*delta_tau*T+8*delta_tau*delta_kappa-6*T*delta_kappa+3*delta_tau2+T2);
  double omega22 = -delta_tau2/T4*(90*delta_kappa2-64*delta_tau*T+120*delta_tau*delta_kappa-96*T*delta_kappa+45*delta_tau2+18*T2);
  double omega23 = delta_tau2/(2*T3)*(30*delta_kappa2-24*delta_tau*T+40*delta_tau*delta_kappa-36*T*delta_kappa+15*delta_tau2+9*T2);

  // Calculate 'lambda' interpolation values
  double lambda12 = delta_tau*delta_kappa3/T4*(t2 - 4*t1 + 3*tau);
  double lambda13 = delta_tau2*delta_kappa3/(2*T3);

  double lambda22 = -(90*delta_tau2*delta_kappa2-56*delta_tau3*T+45*delta_tau4-T4+120*delta_tau3*delta_kappa+12*delta_tau2*T2-84*delta_tau2*T*delta_kappa)/T4;
  double lambda23 = -delta_tau/(2*T3)*(3*delta_tau*T2-16*delta_tau2*T+15*delta_tau3-2*T3+30*delta_tau*delta_kappa2+40*delta_tau2*delta_kappa-24*delta_tau*T*delta_kappa);

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = knot2->getPose()->evaluate()/knot1->getPose()->evaluate();

  // Get se3 algebra of relative matrix
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Intermediate variable
  Eigen::Matrix<double,6,6> varpicurl2 = lgmath::se3::curlyhat(J_21_inv*knot2->getVelocity()->getValue());

  // Calculate interpolated relative se3 algebra
  Eigen::Matrix<double,6,1> xi_i1 = lambda12*knot1->getVelocity()->getValue() +
                                    lambda13*knot1->getAcceleration()->getValue() +
                                    omega11*xi_21 +
                                    omega12*J_21_inv*knot2->getVelocity()->getValue()+
                                    omega13*(-0.5*varpicurl2*knot2->getVelocity()->getValue() + J_21_inv*knot2->getAcceleration()->getValue());

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_t1 = lgmath::se3::vec2jac(xi_i1);

  // Calculate interpolated relative se3 algebra
  Eigen::VectorXd xi_it = J_t1*(lambda22*knot1->getVelocity()->getValue() +
                                lambda23*knot1->getAcceleration()->getValue() +
                                omega21*xi_21 +
                                omega22*J_21_inv*knot2->getVelocity()->getValue() +
                                omega23*(-0.5*varpicurl2*knot2->getVelocity()->getValue() + J_21_inv*knot2->getAcceleration()->getValue()));

   return xi_it;
 }

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
///        should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::addPosePrior(const steam::Time& time,
                                      const lgmath::se3::Transformation& pose,
                                      const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] no knot at provided time.");
  }

  // Get reference
  const SteamCATrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(!knotRef->getPose()->isActive()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] tried to add prior to locked pose.");
  }

  // Set up loss function, noise model, and error function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(cov));
  steam::TransformErrorEval::Ptr errorfunc(new steam::TransformErrorEval(pose, knotRef->getPose()));

  // Create cost term
  posePriorFactor_ = steam::WeightedLeastSqCostTerm<6,6>::Ptr(
        new steam::WeightedLeastSqCostTerm<6,6>(errorfunc, sharedNoiseModel, sharedLossFunc));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary velocity prior factor at a knot time. Note that only a single velocity
///        prior should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::addVelocityPrior(const steam::Time& time,
                                          const Eigen::Matrix<double,6,1>& velocity,
                                          const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const SteamCATrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(knotRef->getVelocity()->isLocked()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] tried to add prior to locked velocity.");
  }

  // Set up loss function, noise model, and error function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(cov));
  steam::VectorSpaceErrorEval<6,6>::Ptr errorfunc(new steam::VectorSpaceErrorEval<6,6>(velocity, knotRef->getVelocity()));

  // Create cost term
  velocityPriorFactor_ = steam::WeightedLeastSqCostTerm<6,6>::Ptr(
        new steam::WeightedLeastSqCostTerm<6,6>(errorfunc, sharedNoiseModel, sharedLossFunc));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary acceleration prior factor at a knot time. Note that only a single acceleration
///        prior should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::addAccelerationPrior(const steam::Time& time,
  const Eigen::Matrix<double,6,1>& acceleration,
  const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
  throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
  throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const SteamCATrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(knotRef->getAcceleration()->isLocked()) {
  throw std::runtime_error("[GpTrajectory][addAccelerationPrior] tried to add prior to locked acceleration.");
  }

  // Set up loss function, noise model, and error function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  steam::BaseNoiseModel<6>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<6>(cov));
  steam::VectorSpaceErrorEval<6,6>::Ptr errorfunc(new steam::VectorSpaceErrorEval<6,6>(acceleration, knotRef->getAcceleration()));

  // Create cost term
  accelerationPriorFactor_ = steam::WeightedLeastSqCostTerm<6,6>::Ptr(
    new steam::WeightedLeastSqCostTerm<6,6>(errorfunc, sharedNoiseModel, sharedLossFunc));
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get cost terms associated with the prior for unlocked parts of the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::appendPriorCostTerms(
    const ParallelizedCostTermCollection::Ptr& costTerms) const {

  // If empty, return none
  if (knotMap_.empty()) {
    return;
  }

  // Check for pose, velocity priors, and acceleration priors
  if (posePriorFactor_) {
    costTerms->add(posePriorFactor_);
  }
  if (velocityPriorFactor_) {
    costTerms->add(velocityPriorFactor_);
  }
  if (accelerationPriorFactor_) {
    costTerms->add(accelerationPriorFactor_);
  }
  // All prior factors will use an L2 loss function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  // steam::GemanMcClureLossFunc::Ptr sharedLossFunc(new steam::GemanMcClureLossFunc(1));

  // Initialize first iterator
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it1 = knotMap_.begin();
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("No knots...");
  }

  // Iterate through all states.. if any are unlocked, supply a prior term
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it2 = it1; ++it2;
  for (; it2 != knotMap_.end(); ++it1, ++it2) {

    // Get knots
    const SteamCATrajVar::ConstPtr& knot1 = it1->second;
    const SteamCATrajVar::ConstPtr& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if(knot1->getPose()->isActive()  || !knot1->getVelocity()->isLocked() ||
       knot2->getPose()->isActive()  || !knot2->getVelocity()->isLocked() ||
       !knot2->getAcceleration()->isLocked()  || !knot2->getAcceleration()->isLocked()) {

      // Generate 18 x 18 information matrix for GP prior factor
      Eigen::Matrix<double,18,18> Qi_inv;
      double one_over_dt = 1.0/(knot2->getTime() - knot1->getTime()).seconds();
      double one_over_dt2 = one_over_dt*one_over_dt;
      double one_over_dt3 = one_over_dt2*one_over_dt;
      double one_over_dt4 = one_over_dt3*one_over_dt;
      double one_over_dt5 = one_over_dt4*one_over_dt;

      Qi_inv.block<6,6>(0,0) = 720.0 * one_over_dt5 * Qc_inv_;
      Qi_inv.block<6,6>(6,6) = 192.0 * one_over_dt3  * Qc_inv_;
      Qi_inv.block<6,6>(12,12) = 9.0 * one_over_dt  * Qc_inv_;
      Qi_inv.block<6,6>(6,0) = Qi_inv.block<6,6>(0,6) = -360.0 * one_over_dt4 * Qc_inv_;
      Qi_inv.block<6,6>(12,0) = Qi_inv.block<6,6>(0,12) = 60.0 * one_over_dt3 * Qc_inv_;
      Qi_inv.block<6,6>(12,6) = Qi_inv.block<6,6>(6,12) = -36.0 * one_over_dt2 * Qc_inv_;
      
      // std::cout << Qi_inv.block<6,6>(12,0) << std::endl << std::endl;
      // std::cout << Qi_inv.block<6,6>(0,12) << std::endl;
      steam::BaseNoiseModelX::Ptr sharedGPNoiseModel(
            new steam::StaticNoiseModelX(Qi_inv, steam::INFORMATION));

      // Create cost term
      steam::se3::SteamCATrajPriorFactor::Ptr errorfunc(
            new steam::se3::SteamCATrajPriorFactor(knot1, knot2));
      steam::WeightedLeastSqCostTermX::Ptr cost(
            new steam::WeightedLeastSqCostTermX(errorfunc, sharedGPNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajInterface::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  // Iterate over trajectory
  std::map<boost::int64_t, SteamCATrajVar::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {

    // Append active states in transform evaluator
    it->second->getPose()->getActiveStateVariables(outStates);

    // Check if velocity is locked
    if (!it->second->getVelocity()->isLocked()) {
      (*outStates)[it->second->getVelocity()->getKey().getID()] = it->second->getVelocity();
    }
  }
}

} // se3
} // steam
