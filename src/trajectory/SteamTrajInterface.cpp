//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamTrajInterface.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/SteamTrajInterface.hpp>

#include <lgmath.hpp>

#include <steam/trajectory/SteamTrajPoseInterpEval.hpp>
#include <steam/trajectory/SteamTrajPriorFactor.hpp>
#include <steam/evaluator/samples/VectorSpaceErrorEval.hpp>

#include <steam/evaluator/blockauto/transform/TransformEvalOperations.hpp>
#include <steam/evaluator/blockauto/transform/ConstVelTransformEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
///        Note, without providing Qc, the trajectory can be used safely for interpolation,
///        but should not be used for estimation.
//////////////////////////////////////////////////////////////////////////////////////////////
SteamTrajInterface::SteamTrajInterface(bool allowExtrapolation) :
  Qc_inv_(Eigen::Matrix<double,6,6>::Identity()), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamTrajInterface::SteamTrajInterface(const Eigen::Matrix<double,6,6>& Qc_inv,
                                       bool allowExtrapolation) :
  Qc_inv_(Qc_inv), allowExtrapolation_(allowExtrapolation) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get pose prior cost
//////////////////////////////////////////////////////////////////////////////////////////////
double SteamTrajInterface::getPosePriorCost() {
  if(posePriorFactor_ != nullptr) {
    return posePriorFactor_->cost();
  } else {
    return 0.0;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get velocity prior cost
//////////////////////////////////////////////////////////////////////////////////////////////
double SteamTrajInterface::getVelocityPriorCost() {
  if(velocityPriorFactor_ != nullptr) {
    return velocityPriorFactor_->cost();
  } else {
    return 0.0;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::add(const SteamTrajVar::Ptr& knot) {

  // Todo, check that time does not already exist in map?

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamTrajVar::Ptr>(knot->getTime().nanosecs(), knot));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::add(const steam::Time& time,
                             const se3::TransformEvaluator::Ptr& T_k0,
                             const VectorSpaceStateVar::Ptr& velocity) {

  // Check velocity input
  if (velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }

  // Todo, check that time does not already exist in map?

  // Make knot
  SteamTrajVar::Ptr newEntry(new SteamTrajVar(time, T_k0, velocity));

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamTrajVar::Ptr>(time.nanosecs(), newEntry));
}

void SteamTrajInterface::add(const steam::Time& time,
                             const se3::TransformEvaluator::Ptr& T_k0,
                             const VectorSpaceStateVar::Ptr& velocity,
                             const Eigen::Matrix<double,12,12> cov) {

  // Check velocity input
  if (velocity->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }

  // Todo, check that time does not already exist in map?

  // Make knot
  SteamTrajVar::Ptr newEntry(new SteamTrajVar(time, T_k0, velocity, cov));

  // Insert in map
  knotMap_.insert(knotMap_.end(),
                  std::pair<boost::int64_t, SteamTrajVar::Ptr>(time.nanosecs(), newEntry));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity,
           const VectorSpaceStateVar::Ptr& acceleration) {
  add(time, T_k0, velocity);
}

void SteamTrajInterface::add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
           const VectorSpaceStateVar::Ptr& velocity,
           const VectorSpaceStateVar::Ptr& acceleration,
           const Eigen::Matrix<double,18,18> cov) {
  add(time, T_k0, velocity, cov.topLeftCorner<12,12>());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
TransformEvaluator::ConstPtr SteamTrajInterface::getInterpPoseEval(const steam::Time& time) const {
  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }
  // Get iterator to first element with time equal to or greater than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());
  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      --it1; // should be safe, as we checked that the map was not empty..
      const SteamTrajVar::Ptr& endKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(endKnot->getVelocity(), time - endKnot->getTime());
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

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      const SteamTrajVar::Ptr& startKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(startKnot->getVelocity(),
                                                 time - startKnot->getTime());
      return compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }
  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }
  // Create interpolated evaluator
  return SteamTrajPoseInterpEval::MakeShared(time, it1->second, it2->second);
}

Eigen::VectorXd SteamTrajInterface::getVelocity(const steam::Time& time) {
  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or greater than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
     = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

   // If we allow extrapolation, return constant-velocity interpolated entry
   if (allowExtrapolation_) {
     --it1; // should be safe, as we checked that the map was not empty..
     const SteamTrajVar::Ptr& endKnot = it1->second;
     return endKnot->getVelocity()->getValue();
   } else {
     throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
   }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
     const SteamTrajVar::Ptr& knot = it1->second;
     // return state variable exactly (no interp)
     return knot->getVelocity()->getValue();
  }

  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {
    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
     const SteamTrajVar::Ptr& startKnot = it1->second;
     return startKnot->getVelocity()->getValue();
    } else {
     throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                            "should not trigger... report to a STEAM contributor.");
  }

  // OK, we actually need to interpolate.
  // Follow a similar setup to SteamTrajPoseInterpEval

  // Convenience defs
  auto &knot1 = it1->second;
  auto &knot2 = it2->second;

  // Calculate time constants
  double tau = (time - knot1->getTime()).seconds();
  double T = (knot2->getTime() - knot1->getTime()).seconds();
  double ratio = tau/T;
  double ratio2 = ratio*ratio;
  double ratio3 = ratio2*ratio;

  // Calculate 'psi' interpolation values
  double psi11 = 3.0*ratio2 - 2.0*ratio3;
  double psi12 = tau*(ratio2 - ratio);
  double psi21 = 6.0*(ratio - ratio2)/T;
  double psi22 = 3.0*ratio2 - 2.0*ratio;

  // Calculate (some of the) 'lambda' interpolation values
  double lambda12 = tau - T*psi11 - psi12;
  double lambda22 = 1.0 - T*psi21 - psi22;

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = knot2->getPose()->evaluate()/knot1->getPose()->evaluate();

  // Get se3 algebra of relative matrix (and cache it)
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();

  // Calculate the 6x6 associated Jacobian (and cache it)
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Calculate interpolated relative se3 algebra
  Eigen::Matrix<double,6,1> xi_i1 = lambda12*knot1->getVelocity()->getValue() +
                                   psi11*xi_21 +
                                   psi12*J_21_inv*knot2->getVelocity()->getValue();

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_t1 = lgmath::se3::vec2jac(xi_i1);

  // Calculate interpolated relative se3 algebra
  Eigen::VectorXd xi_it = J_t1*(lambda22*knot1->getVelocity()->getValue() +
                                   psi21*xi_21 +
                                   psi22*J_21_inv*knot2->getVelocity()->getValue()
                                   );

   return xi_it;
 }

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a unary pose prior factor at a knot time. Note that only a single pose prior
///        should exist on a trajectory, adding a second will overwrite the first.
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::addPosePrior(const steam::Time& time,
                                      const lgmath::se3::Transformation& pose,
                                      const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addPosePrior] no knot at provided time.");
  }

  // Get reference
  const SteamTrajVar::Ptr& knotRef = it->second;

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
void SteamTrajInterface::addVelocityPrior(const steam::Time& time,
                                          const Eigen::Matrix<double,6,1>& velocity,
                                          const Eigen::Matrix<double,6,6>& cov) {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] map was empty.");
  }

  // Try to find knot at same time
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it = knotMap_.find(time.nanosecs());
  if (it == knotMap_.end()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] no knot at provided time.");
  }

  // Get reference
  const SteamTrajVar::Ptr& knotRef = it->second;

  // Check that the pose is not locked
  if(knotRef->getVelocity()->isLocked()) {
    throw std::runtime_error("[GpTrajectory][addVelocityPrior] tried to add prior to locked pose.");
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
/// \brief Get cost terms associated with the prior for unlocked parts of the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::appendPriorCostTerms(
    const ParallelizedCostTermCollection::Ptr& costTerms) const {

  // If empty, return none
  if (knotMap_.empty()) {
    return;
  }

  // Check for pose or velocity priors
  if (posePriorFactor_) {
    costTerms->add(posePriorFactor_);
  }
  if (velocityPriorFactor_) {
    costTerms->add(velocityPriorFactor_);
  }

  // All prior factors will use an L2 loss function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Initialize first iterator
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1 = knotMap_.begin();
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("No knots...");
  }

  // Iterate through all states.. if any are unlocked, supply a prior term
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it2 = it1; ++it2;
  for (; it2 != knotMap_.end(); ++it1, ++it2) {

    // Get knots
    const SteamTrajVar::ConstPtr& knot1 = it1->second;
    const SteamTrajVar::ConstPtr& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if(knot1->getPose()->isActive()  || !knot1->getVelocity()->isLocked() ||
       knot2->getPose()->isActive()  || !knot2->getVelocity()->isLocked() ) {

      // Generate 12 x 12 information matrix for GP prior factor
      Eigen::Matrix<double,12,12> Qi_inv;
      double one_over_dt = 1.0/(knot2->getTime() - knot1->getTime()).seconds();
      double one_over_dt2 = one_over_dt*one_over_dt;
      double one_over_dt3 = one_over_dt2*one_over_dt;
      Qi_inv.block<6,6>(0,0) = 12.0 * one_over_dt3 * Qc_inv_;
      Qi_inv.block<6,6>(6,0) = Qi_inv.block<6,6>(0,6) = -6.0 * one_over_dt2 * Qc_inv_;
      Qi_inv.block<6,6>(6,6) = 4.0 * one_over_dt  * Qc_inv_;
      steam::BaseNoiseModelX::Ptr sharedGPNoiseModel(
            new steam::StaticNoiseModelX(Qi_inv, steam::INFORMATION));

      // Create cost term
      steam::se3::SteamTrajPriorFactor::Ptr errorfunc(
            new steam::se3::SteamTrajPriorFactor(knot1, knot2));
      steam::WeightedLeastSqCostTermX::Ptr cost(
            new steam::WeightedLeastSqCostTermX(errorfunc, sharedGPNoiseModel, sharedLossFunc));
      costTerms->add(cost);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Store solver in trajectory, needed for querying covariances later
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::setSolver(std::shared_ptr<GaussNewtonSolverBase> solver) {
  solver_ = solver;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get interpolated/extrapolated covariance at given time, for now only support at knot times
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::getCovariance(const steam::Time& time) const {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {
    --it1;
    // If we allow extrapolation
    // if (allowExtrapolation_) {
    //   return extrapCovariance(solver, time);
    // } else {
    //   throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    // }
  }

  // Check if we requested time exactly
  // if (it1->second->getTime() == time) {
    // check if the state variable at time has associated covariacne
    if (it1->second->covarianceSet()) {
      return it1->second->getCovariance();
    }  
    // return covariance exactly (no interp)
    std::map<unsigned int, steam::StateVariableBase::Ptr> outState;
    it1->second->getPose()->getActiveStateVariables(&outState);
    
    std::vector<steam::StateKey> keys;
    keys.push_back(outState.begin()->second->getKey());
    keys.push_back(it1->second->getVelocity()->getKey());

    steam::BlockMatrix covariance = solver_->queryCovarianceBlock(keys);

    Eigen::Matrix<double,12,12> output;
    output.block<6,6>(0,0) = covariance.copyAt(0,0);
    output.block<6,6>(0,6) = covariance.copyAt(0,1);
    output.block<6,6>(6,0) = covariance.copyAt(1,0);
    output.block<6,6>(6,6) = covariance.copyAt(1,1);
    return output;
  // }

  // // TODO(DAVID): Be able to handle locked states
  // // Check if state is locked
  // std::map<unsigned int, steam::StateVariableBase::Ptr> states;
  // it1->second->getPose()->getActiveStateVariables(&states);
  // if (states.size() == 0) {
  //   throw std::runtime_error("Attempted covariance interpolation with locked states");
  // }

  // // TODO(DAVID): Extrapolation behind first entry
  
  // // Check if we requested before first time
  // if (it1 == knotMap_.begin()) {

  //   // If we allow extrapolation, return constant-velocity interpolated entry
  //   if (allowExtrapolation_) {
  //     const SteamTrajVar::Ptr& startKnot = it1->second;
  //     TransformEvaluator::Ptr T_t_k =
  //         ConstVelTransformEvaluator::MakeShared(startKnot->getVelocity(),
  //                                                time - startKnot->getTime());
  //     return compose(T_t_k, startKnot->getPose());
  //   } else {
  //     throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
  //   }
  // }
  

  // return interpCovariance(solver, time);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  // Iterate over trajectory
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {

    // Append active states in transform evaluator
    it->second->getPose()->getActiveStateVariables(outStates);

    // Check if velocity is locked
    if (!it->second->getVelocity()->isLocked()) {
      (*outStates)[it->second->getVelocity()->getKey().getID()] = it->second->getVelocity();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get interpolated/extrapolated covariance at given time
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::getCovariance(GaussNewtonSolverBase& solver,
    const steam::Time& time) const {

  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getEvaluator] map was empty");
  }

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // Check if time is passed the last entry
  if (it1 == knotMap_.end()) {

    // If we allow extrapolation
    if (allowExtrapolation_) {
      return extrapCovariance(solver, time);
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }

  // Check if we requested time exactly
  if (it1->second->getTime() == time) {
    // return covariance exactly (no interp)
    std::map<unsigned int, steam::StateVariableBase::Ptr> outState;
    it1->second->getPose()->getActiveStateVariables(&outState);
    
    std::vector<steam::StateKey> keys;
    keys.push_back(outState.begin()->second->getKey());
    keys.push_back(it1->second->getVelocity()->getKey());

    steam::BlockMatrix covariance = solver.queryCovarianceBlock(keys);

    Eigen::Matrix<double,12,12> output;
    output.block<6,6>(0,0) = covariance.copyAt(0,0);
    output.block<6,6>(0,6) = covariance.copyAt(0,1);
    output.block<6,6>(6,0) = covariance.copyAt(1,0);
    output.block<6,6>(6,6) = covariance.copyAt(1,1);
    return output;
  }

  // TODO(DAVID): Be able to handle locked states
  // Check if state is locked
  std::map<unsigned int, steam::StateVariableBase::Ptr> states;
  it1->second->getPose()->getActiveStateVariables(&states);
  if (states.size() == 0) {
    throw std::runtime_error("Attempted covariance interpolation with locked states");
  }

  // TODO(DAVID): Extrapolation behind first entry
  /*
  // Check if we requested before first time
  if (it1 == knotMap_.begin()) {

    // If we allow extrapolation, return constant-velocity interpolated entry
    if (allowExtrapolation_) {
      const SteamTrajVar::Ptr& startKnot = it1->second;
      TransformEvaluator::Ptr T_t_k =
          ConstVelTransformEvaluator::MakeShared(startKnot->getVelocity(),
                                                 time - startKnot->getTime());
      return compose(T_t_k, startKnot->getPose());
    } else {
      throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
    }
  }
  */

  return interpCovariance(solver, time);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Covariance translation from global(2x2 block matrix, each 6x6) to local
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::translateCovToLocal(const steam::BlockMatrix& global_cov, 
    const SteamTrajVar::ConstPtr& knot1, const SteamTrajVar::ConstPtr& knot2) const {

  Eigen::Matrix<double,12,12> P_11;
  Eigen::Matrix<double,12,12> P_12;
  Eigen::Matrix<double,12,12> P_22;

  P_11.block<6,6>(0,0) = global_cov.copyAt(0,0);
  P_11.block<6,6>(0,6) = global_cov.copyAt(0,1);
  P_11.block<6,6>(6,0) = global_cov.copyAt(1,0);
  P_11.block<6,6>(6,6) = global_cov.copyAt(1,1);

  P_12.block<6,6>(0,0) = global_cov.copyAt(0,2);
  P_12.block<6,6>(0,6) = global_cov.copyAt(0,3);
  P_12.block<6,6>(6,0) = global_cov.copyAt(1,2);
  P_12.block<6,6>(6,6) = global_cov.copyAt(1,3);

  P_22.block<6,6>(0,0) = global_cov.copyAt(2,2);
  P_22.block<6,6>(0,6) = global_cov.copyAt(2,3);
  P_22.block<6,6>(6,0) = global_cov.copyAt(3,2);
  P_22.block<6,6>(6,6) = global_cov.copyAt(3,3);

  // Create constants
  Eigen::Matrix<double,12,12> Gam1 = Eigen::MatrixXd::Zero(12,12);
  Eigen::Matrix<double,12,12> Gam2 = Eigen::MatrixXd::Zero(12,12);
  Eigen::Matrix<double,12,12> Xi1 = Eigen::MatrixXd::Zero(12,12);
  Eigen::Matrix<double,12,12> Xi2 = Eigen::MatrixXd::Zero(12,12);

  Eigen::Matrix<double,6,6> J_11_inv = lgmath::se3::vec2jacinv(Eigen::MatrixXd::Identity(6,6));
  Gam1.block<6,6>(0,0) = J_11_inv;
  Gam1.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(knot1->getVelocity()->getValue())*J_11_inv;
  Gam1.block<6,6>(6,6) = J_11_inv;

  lgmath::se3::Transformation T_21 = knot2->getPose()->evaluate()/knot1->getPose()->evaluate();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(T_21.vec());
  Gam2.block<6,6>(0,0) = J_21_inv;
  Gam2.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(knot2->getVelocity()->getValue())*J_21_inv;
  Gam2.block<6,6>(6,6) = J_21_inv;

  Xi1.block<6,6>(0,0) = Eigen::MatrixXd::Identity(6,6);
  Xi2.block<6,6>(0,0) = lgmath::se3::tranAd(T_21.matrix());

  // Translate
  Eigen::Matrix<double,12,12> P_11_local = Gam1*(P_11 - Xi1*P_11*Xi1.transpose())*Gam1.transpose();
  Eigen::Matrix<double,12,12> P_12_local = Gam1*(P_12 - Xi1*P_11*Xi2.transpose())*Gam2.transpose();
  Eigen::Matrix<double,12,12> P_22_local = Gam2*(P_22 - Xi2*P_11*Xi2.transpose())*Gam2.transpose();

  // Return
  Eigen::Matrix<double,24,24> output;
  output.block<12,12>(0,0)   = P_11_local;
  output.block<12,12>(0,12)  = P_12_local;
  output.block<12,12>(12,0)  = P_12_local.transpose();
  output.block<12,12>(12,12) = P_22_local;
  return output;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Covariance translation from global(6x6 matrix, meaning 1 state) to local
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::translateCovToLocal(const Eigen::MatrixXd& global_cov, 
    const SteamTrajVar::ConstPtr& knot1) const {

  // Create constants
  Eigen::Matrix<double,12,12> Gam = Eigen::MatrixXd::Zero(12,12);
  Eigen::Matrix<double,6,6> J_11_inv = lgmath::se3::vec2jacinv(Eigen::MatrixXd::Identity(6,6));
  Gam.block<6,6>(0,0) = J_11_inv;
  Gam.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(knot1->getVelocity()->getValue())*J_11_inv;
  Gam.block<6,6>(6,6) = J_11_inv;

  Eigen::Matrix<double,12,12> Xi = Eigen::MatrixXd::Zero(12,12);
  Xi.block<6,6>(0,0) = Eigen::MatrixXd::Identity(6,6);

  // Translate and return
  return Gam*(global_cov - Xi*global_cov*Xi.transpose())*Gam.transpose();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Covariance translation of interpolated covariance from local to global
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::translateCovToGlobal(const Eigen::MatrixXd& local_cov,
    const Eigen::MatrixXd& global_frame_cov, const lgmath::se3::Transformation& local_pose, 
    const Eigen::MatrixXd& velocity) const {

  // Create constants. 'tau' is time index interpolated between '1' and '2'. If extrapolation is
  // the case, 'tau' is time index extrapolated past index '1'.
  Eigen::Matrix<double,6,6> J_tau1 = lgmath::se3::vec2jac(local_pose.vec());
  Eigen::Matrix<double,12,12> Gam_inv = Eigen::MatrixXd::Zero(12,12);
  Gam_inv.block<6,6>(0,0) = J_tau1;
  Gam_inv.block<6,6>(6,0) = -0.5*J_tau1*lgmath::se3::curlyhat(velocity);
  Gam_inv.block<6,6>(6,6) = J_tau1;

  Eigen::Matrix<double,12,12> Xi = Eigen::MatrixXd::Zero(12,12);
  Xi.block<6,6>(0,0) = lgmath::se3::tranAd(local_pose.matrix());

  return Gam_inv*local_cov*Gam_inv.transpose() + Xi*global_frame_cov*Xi.transpose();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get interpolated state at given time
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamTrajInterface::interpState(lgmath::se3::Transformation* pose, Eigen::MatrixXd* velocity,
    const Eigen::MatrixXd& lambda_psi, const steam::Time& time,
    const SteamTrajVar::ConstPtr& knot1, const SteamTrajVar::ConstPtr& knot2) const {

  // Interpolated state, which includes velocity is required for covariance interpolation.
  // SteamTrajPoseInterpEval only interpolates the pose.

  // Constants
  Eigen::Matrix<double,12,12> Lambda = lambda_psi.block<12,12>(0,0);
  Eigen::Matrix<double,12,12> Psi = lambda_psi.block<12,12>(0,12);

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = knot2->getPose()->evaluate()/knot1->getPose()->evaluate();

  // Get se3 algebra of relative matrix
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Calculate interpolated relative se3 algebra
  double psi11 = Psi(0,0);
  double psi12 = Psi(0,6);
  double lambda12 = Lambda(0,6);
  Eigen::Matrix<double,6,1> xi_i1 = lambda12*knot1->getVelocity()->getValue() +
                                    psi11*xi_21 +
                                    psi12*J_21_inv*knot2->getVelocity()->getValue();

  // Calculate interpolated relative transformation matrix
  lgmath::se3::Transformation T_i1(xi_i1);

  // Global interpolated pose
  *pose = T_i1*knot1->getPose()->evaluate();

  // Note(DAVID): Computation above is directly from SteamTrajPoseInterpEval. I'm unsure of
  // Sean's notation, particularly if xi is referring to the greek letter or x indexed with i.
  // It does seem that i after an underscore is referring to the interpolated time index.

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_i1 = lgmath::se3::vec2jac(xi_i1);

  // TODO(DAVID): As how Sean simplified the pose interpolation, do the same for velocity.
  // i.e. do not fully compute the matrix multiplications.

  // Local state variables. i.e. gam1 -> gamma_k(t_k)
  Eigen::Matrix<double,12,1> gam1 = Eigen::MatrixXd::Zero(12,1);
  gam1.block<6,1>(6,0) = knot1->getVelocity()->getValue();

  Eigen::Matrix<double,12,1> gam2 = Eigen::MatrixXd::Zero(12,1);
  gam2.block<6,1>(0,0) = xi_21;
  gam2.block<6,1>(6,0) = J_21_inv*knot2->getVelocity()->getValue();

  *velocity = J_i1*(Lambda.block<6,12>(6,0)*gam1 + Psi.block<6,12>(6,0)*gam2);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute covariance interpolation at given time
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::interpCovariance(GaussNewtonSolverBase& solver, 
    const steam::Time& time) const {

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  // Get iterators bounding the time interval
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->getTime() || time >= it2->second->getTime()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // Get required pose/velocity keys
  // TODO(DAVID): Find a better way to get pose keys...
  std::vector<steam::StateKey> keys;
  {
    std::map<unsigned int, steam::StateVariableBase::Ptr> outState1;
    it1->second->getPose()->getActiveStateVariables(&outState1);

    std::map<unsigned int, steam::StateVariableBase::Ptr> outState2;
    it2->second->getPose()->getActiveStateVariables(&outState2);

    keys.push_back(outState1.begin()->second->getKey());
    keys.push_back(it1->second->getVelocity()->getKey());
    keys.push_back(outState2.begin()->second->getKey());
    keys.push_back(it2->second->getVelocity()->getKey());
  }

  // Get required covariances using the keys
  steam::BlockMatrix global_cov = solver.queryCovarianceBlock(keys);

  // Approximately translate global covariances (P_11 ... P_22) to local frame 1
  Eigen::Matrix<double,24,24> local_cov = translateCovToLocal(global_cov,
      it1->second, it2->second);

  // Compute constants (Refer to eq 6.116 in Sean Anderson's thesis, note psi here is omega in
  // his thesis)
  Eigen::Matrix<double,12,24> Lambda_Psi; // [Lambda  Psi] block matrix
  Eigen::Matrix<double,12,12> Q_check;
  { // TODO(DAVID): First part is copied from SteamTrajPoseInterpEval. Look into a better way?
    double tau = (time - it1->second->getTime()).seconds();
    double T = (it2->second->getTime() - it1->second->getTime()).seconds();
    double ratio = tau/T;
    double ratio2 = ratio*ratio;
    double ratio3 = ratio2*ratio;

    // Calculate 'psi' interpolation values
    double psi11 = 3.0*ratio2 - 2.0*ratio3;
    double psi12 = tau*(ratio2 - ratio);
    double psi21 = 6.0*(ratio - ratio2)/T;
    double psi22 = 3.0*ratio2 - 2.0*ratio;

    // Calculate 'lambda' interpolation values
    double lambda11 = 1.0 - psi11;
    double lambda12 = tau - T*psi11 - psi12;
    double lambda21 = -psi21;
    double lambda22 = 1.0 - psi21 - psi22;

    // Formulate Lambda and Psi matrices
    Eigen::Matrix<double,6,6> identity = Eigen::MatrixXd::Identity(6,6);
    Eigen::Matrix<double,12,12> Lambda;
    Lambda.block<6,6>(0,0) = lambda11*identity;
    Lambda.block<6,6>(0,6) = lambda12*identity;
    Lambda.block<6,6>(6,0) = lambda21*identity;
    Lambda.block<6,6>(6,6) = lambda22*identity;

    Eigen::Matrix<double,12,12> Psi;
    Psi.block<6,6>(0,0) = psi11*identity;
    Psi.block<6,6>(0,6) = psi12*identity;
    Psi.block<6,6>(6,0) = psi21*identity;
    Psi.block<6,6>(6,6) = psi22*identity;

    Lambda_Psi.block<12,12>(0,0) = Lambda;
    Lambda_Psi.block<12,12>(0,12) = Psi;

    // Formulate Q_check
    Eigen::Matrix<double,6,6> Qc = Qc_inv_.inverse();

    Eigen::Matrix<double,12,12> Qk_tau;
    Qk_tau.block<6,6>(0,0) = 1.0/3.0*tau*tau*tau*Qc;
    Qk_tau.block<6,6>(0,6) = 0.5*tau*tau*Qc;
    Qk_tau.block<6,6>(6,0) = Qk_tau.block<6,6>(0,6);
    Qk_tau.block<6,6>(6,6) = tau*Qc;

    Eigen::Matrix<double,12,12> Qk_2_inv;
    double T_inv = 1.0/T;
    Qk_2_inv.block<6,6>(0,0) = 12.0*T_inv*T_inv*T_inv*Qc_inv_;
    Qk_2_inv.block<6,6>(0,6) = -6.0*T_inv*T_inv*Qc_inv_;
    Qk_2_inv.block<6,6>(6,0) = Qk_2_inv.block<6,6>(0,6);
    Qk_2_inv.block<6,6>(6,6) = 4.0*T_inv*Qc_inv_;

    Eigen::Matrix<double,12,12> tran_2_tau = Eigen::MatrixXd::Identity(12,12); // Phi in lit.
    tran_2_tau.block<6,6>(0,6) = (T - tau)*Eigen::MatrixXd::Identity(6,6);  

    Q_check = Qk_tau - Qk_tau*tran_2_tau.transpose()*Qk_2_inv.transpose()*
        tran_2_tau*Qk_tau.transpose();
  }

  // Interpolate covariance in local frame
  Eigen::MatrixXd local_interp = Lambda_Psi*local_cov*Lambda_Psi.transpose() + Q_check;

  // Need interpolated state
  lgmath::se3::Transformation pose;
  Eigen::MatrixXd velocity;
  interpState(&pose, &velocity, Lambda_Psi, time, it1->second, it2->second);

  // Approximately translate result to global frame and return
  Eigen::Matrix<double,12,12> P_11;

  P_11.block<6,6>(0,0) = global_cov.copyAt(0,0);
  P_11.block<6,6>(0,6) = global_cov.copyAt(0,1);
  P_11.block<6,6>(6,0) = global_cov.copyAt(1,0);
  P_11.block<6,6>(6,6) = global_cov.copyAt(1,1);

  return translateCovToGlobal(local_interp, P_11, 
      pose/it1->second->getPose()->evaluate(), velocity);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute covariance interpolation at given time
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::extrapCovariance(GaussNewtonSolverBase& solver, 
    const steam::Time& time) const {

  // Get iterator to first element with time equal to or great than 'time'
  std::map<boost::int64_t, SteamTrajVar::Ptr>::const_iterator it1
      = knotMap_.lower_bound(time.nanosecs());

  --it1; // should be safe, as we checked that the map was not empty..
  // const SteamTrajVar::Ptr& endKnot = it1->second;

  // Get required pose/velocity keys
  // TODO(DAVID): Find a better way to get pose keys...
  std::vector<steam::StateKey> keys;
  {
    std::map<unsigned int, steam::StateVariableBase::Ptr> outState1;
    it1->second->getPose()->getActiveStateVariables(&outState1);

    keys.push_back(outState1.begin()->second->getKey());
    keys.push_back(it1->second->getVelocity()->getKey());
  }

  // Get required covariances using the keys
  steam::BlockMatrix global_cov = solver.queryCovarianceBlock(keys);
  Eigen::Matrix<double,12,12> P_11;

  P_11.block<6,6>(0,0) = global_cov.copyAt(0,0);
  P_11.block<6,6>(0,6) = global_cov.copyAt(0,1);
  P_11.block<6,6>(6,0) = global_cov.copyAt(1,0);
  P_11.block<6,6>(6,6) = global_cov.copyAt(1,1);

  // Approximately translate covariance of last state to local frame
  Eigen::MatrixXd local_frame_cov = translateCovToLocal(P_11, it1->second);

  // Extrapolate in local frame
  double tau = (time - it1->second->getTime()).seconds();
  Eigen::Matrix<double,6,6> Qc = Qc_inv_.inverse();

  Eigen::Matrix<double,12,12> Qk_tau;
  Qk_tau.block<6,6>(0,0) = 1.0/3.0*tau*tau*tau*Qc;
  Qk_tau.block<6,6>(0,6) = 0.5*tau*tau*Qc;
  Qk_tau.block<6,6>(6,0) = Qk_tau.block<6,6>(0,6);
  Qk_tau.block<6,6>(6,6) = tau*Qc;

  // Note: This is the transformation matrix, Phi in literature.
  Eigen::Matrix<double,12,12> tran_tau_1 = Eigen::MatrixXd::Identity(12,12);
  tran_tau_1.block<6,6>(0,6) = tau*Eigen::MatrixXd::Identity(6,6);

  Eigen::MatrixXd local_extrap = tran_tau_1*local_frame_cov*tran_tau_1.transpose() + Qk_tau;

  // Extrapolate state. Velocity stays constant
  const SteamTrajVar::Ptr& endKnot = it1->second;
  TransformEvaluator::Ptr T_t_k =
      ConstVelTransformEvaluator::MakeShared(endKnot->getVelocity(), time - endKnot->getTime());

  // Approximately translate to global frame and return
  return translateCovToGlobal(local_extrap, P_11, T_t_k->evaluate(), 
      it1->second->getVelocity()->getValue());
}

} // se3
} // steam
