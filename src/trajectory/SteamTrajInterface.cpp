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
#include <steam/evaluator/samples/TrajErrorEval.hpp>

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
/// \brief Get interpolated/extrapolated covariance on a relative pose between t_a and t_b
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd SteamTrajInterface::getRelativeCovariance(const steam::Time& time_a, const steam::Time& time_b) const {
  // todo (Ben): refactor so less code duplication

  if (solver_ == nullptr) {
    throw std::runtime_error("[GpTrajectory][getRelativeCovariance] solver not set");
  }
  // Check that map is not empty
  if (knotMap_.empty()) {
    throw std::runtime_error("[GpTrajectory][getRelativeCovariance] map was empty");
  }
  if (time_b <= time_a) {
    throw std::runtime_error("[GpTrajectory][getRelativeCovariance] time_b precedes time_a");
  }

  // Get iterators bounding the times
  auto it2a = knotMap_.lower_bound(time_a.nanosecs());
  auto it1a = it2a; it1a--;
  auto it2b = knotMap_.lower_bound(time_b.nanosecs());
  auto it1b = it2b; it1b--;

  if (it2b == knotMap_.end()) {
    // extrapolating
    if (it2a == knotMap_.end()) {
      // unary bracket extrapolation case
      std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1a;
      it1a->second->getPose()->getActiveStateVariables(&out_state_1a);
      if (out_state_1a.empty()) {
        throw std::runtime_error("Attempted covariance interpolation with locked states");
      }

      std::vector<steam::StateKey> keys{out_state_1a.begin()->second->getKey(),
                                        it1a->second->getVelocity()->getKey()};
      steam::BlockMatrix Cov_quad = solver_->queryCovarianceBlock(keys);  // 12 x 12

      // add our 3 state variables
      std::vector<SteamTrajVar> traj_states;
      std::vector<TransformStateVar::Ptr> statevars;

      TransformStateVar::Ptr tmp_state_1a(new TransformStateVar(it1a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_1a = TransformStateEvaluator::MakeShared(tmp_state_1a);
      VectorSpaceStateVar::Ptr tmp_vel_1a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_1a(it1a->second->getTime(), tmp_pose_1a, tmp_vel_1a);
      statevars.push_back(tmp_state_1a);
      traj_states.push_back(tmp_1a);

      TransformStateVar::Ptr tmp_state_a(new TransformStateVar(it1a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_a = TransformStateEvaluator::MakeShared(tmp_state_a);
      VectorSpaceStateVar::Ptr tmp_vel_a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_a(time_a, tmp_pose_a, tmp_vel_a);
      statevars.push_back(tmp_state_a);
      traj_states.push_back(tmp_a);

      TransformStateVar::Ptr tmp_state_b(new TransformStateVar(it1b->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_b = TransformStateEvaluator::MakeShared(tmp_state_b);
      VectorSpaceStateVar::Ptr tmp_vel_b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
      SteamTrajVar tmp_b(time_b, tmp_pose_b, tmp_vel_b);
      statevars.push_back(tmp_state_b);
      traj_states.push_back(tmp_b);

      std::shared_ptr<steam::OptimizationProblem> problem;
      problem.reset(new steam::OptimizationProblem());

      for (auto & state : statevars) {
        problem->addStateVariable(state);
      }
      for (auto & state : traj_states) {
        problem->addStateVariable(state.getVelocity());
      }

      steam::ParallelizedCostTermCollection::Ptr cost_terms;
      cost_terms.reset(new steam::ParallelizedCostTermCollection());

      // one trajectory in this case
      steam::se3::SteamTrajInterface traj_a(Qc_inv_, true);
      for (int i = 0; i < 3; ++i) {
        traj_a.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
      }
      traj_a.appendPriorCostTerms(cost_terms);

      // copy over posterior to
      Eigen::MatrixXd post_cov = Eigen::MatrixXd::Identity(12, 12);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          post_cov.block<6,6>(6*i, 6*j) = Cov_quad.at(i, j);
        }
      }

      steam::BaseNoiseModel<Eigen::Dynamic>::Ptr noise_model(new steam::StaticNoiseModel<Eigen::Dynamic>(post_cov));
      steam::L2LossFunc::Ptr loss_function(new steam::L2LossFunc());

      // Set up trajectory error factor for posterior
      std::vector<se3::TransformEvaluator::Ptr> poses;
      std::vector<VectorSpaceStateVar::Ptr> vels;
      poses.push_back(traj_states[0].getPose());
      vels.push_back(traj_states[0].getVelocity());

      steam::TrajErrorEval::Ptr traj_error(new steam::TrajErrorEval(poses, vels));

      auto traj_factor = steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>::Ptr(
          new steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>(
              traj_error,
              noise_model,
              loss_function));
      cost_terms->add(traj_factor);

      problem->addCostTerm(cost_terms);

      // setup solver and optimize
      std::shared_ptr<steam::GaussNewtonSolverBase> gn_solver;
      steam::DoglegGaussNewtonSolver::Params params;
      params.maxIterations = 1;
      gn_solver.reset(new steam::DoglegGaussNewtonSolver(problem.get(), params));

      try {
        gn_solver->optimize();
      } catch (steam::unsuccessful_step &e) {
        std::cout
            << "Steam has failed to optimize interpolated covariance problem! This is an ERROR."
            << std::endl;
        return Eigen::Matrix<double, 6, 6>::Identity();
      } catch (steam::decomp_failure &e) {
        // Should not occur frequently
        std::cout
            << "Steam has encountered an LL^T decomposition error while optimizing for interpolated covariance! This is an ERROR."
            << std::endl;
        return Eigen::Matrix<double, 6, 6>::Identity();
      }

      std::vector<steam::StateKey> pose_keys{statevars[1]->getKey(), statevars[2]->getKey()};
      auto Cov_a0a0_b0b0 = gn_solver->queryCovarianceBlock(pose_keys);
      lgmath::se3::Transformation T_a = getInterpPoseEval(time_a)->evaluate();
      lgmath::se3::Transformation T_b = getInterpPoseEval(time_b)->evaluate();
      lgmath::se3::Transformation T_b_a = T_b / T_a;

      auto Tadj_b_a = T_b_a.adjoint();
      auto correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
      auto Cov_ba_ba =
          Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
              Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
      std::cout << "Warning: Extrapolate relative covariance (unary) not fully tested yet." << std::endl;   // temporary
      return Cov_ba_ba;

    } else if (it2a->second->getTime() == it1b->second->getTime()) {
      // binary bracket extrapolation case
      std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1a;
      std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_2a;
      it1a->second->getPose()->getActiveStateVariables(&out_state_1a);
      it2a->second->getPose()->getActiveStateVariables(&out_state_2a);
      if (out_state_1a.empty() || out_state_2a.empty()) {
        // TODO: Be able to handle locked states
        throw std::runtime_error("Attempted covariance interpolation with locked states");
      }

      std::vector<steam::StateKey> keys{out_state_1a.begin()->second->getKey(),
                                        it1a->second->getVelocity()->getKey(),
                                        out_state_2a.begin()->second->getKey(),
                                        it2a->second->getVelocity()->getKey()};
      steam::BlockMatrix Cov_quad = solver_->queryCovarianceBlock(keys);  // 24 x 24

      // add our 4 state variables
      std::vector<SteamTrajVar> traj_states;
      std::vector<TransformStateVar::Ptr> statevars;

      TransformStateVar::Ptr tmp_state_1a(new TransformStateVar(it1a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_1a = TransformStateEvaluator::MakeShared(tmp_state_1a);
      VectorSpaceStateVar::Ptr tmp_vel_1a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_1a(it1a->second->getTime(), tmp_pose_1a, tmp_vel_1a);
      statevars.push_back(tmp_state_1a);
      traj_states.push_back(tmp_1a);

      TransformStateVar::Ptr tmp_state_a(new TransformStateVar(it1a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_a = TransformStateEvaluator::MakeShared(tmp_state_a);
      VectorSpaceStateVar::Ptr tmp_vel_a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_a(time_a, tmp_pose_a, tmp_vel_a);
      statevars.push_back(tmp_state_a);
      traj_states.push_back(tmp_a);

      TransformStateVar::Ptr tmp_state_2a(new TransformStateVar(it2a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_2a = TransformStateEvaluator::MakeShared(tmp_state_2a);
      VectorSpaceStateVar::Ptr tmp_vel_2a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it2a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_2a(it2a->second->getTime(), tmp_pose_2a, tmp_vel_2a);
      statevars.push_back(tmp_state_2a);
      traj_states.push_back(tmp_2a);

      TransformStateVar::Ptr tmp_state_b(new TransformStateVar(it1b->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_b = TransformStateEvaluator::MakeShared(tmp_state_b);
      VectorSpaceStateVar::Ptr tmp_vel_b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
      SteamTrajVar tmp_b(time_b, tmp_pose_b, tmp_vel_b);
      statevars.push_back(tmp_state_b);
      traj_states.push_back(tmp_b);

      std::shared_ptr<steam::OptimizationProblem> problem;
      problem.reset(new steam::OptimizationProblem());

      for (auto & state : statevars) {
        problem->addStateVariable(state);
      }
      for (auto & state : traj_states) {
        problem->addStateVariable(state.getVelocity());
      }

      steam::ParallelizedCostTermCollection::Ptr cost_terms;
      cost_terms.reset(new steam::ParallelizedCostTermCollection());

      // one trajectory in this case
      steam::se3::SteamTrajInterface traj_a(Qc_inv_, true);
      for (int i = 0; i < 4; ++i) {
        traj_a.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
      }

      traj_a.appendPriorCostTerms(cost_terms);

      // copy over posterior to
      Eigen::MatrixXd post_cov = Eigen::MatrixXd::Identity(24, 24);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          post_cov.block<6,6>(6*i, 6*j) = Cov_quad.at(i, j);
        }
      }

      steam::BaseNoiseModel<Eigen::Dynamic>::Ptr noise_model(new steam::StaticNoiseModel<Eigen::Dynamic>(post_cov));
      steam::L2LossFunc::Ptr loss_function(new steam::L2LossFunc());

      // Set up trajectory error factor for posterior
      std::vector<se3::TransformEvaluator::Ptr> poses;
      std::vector<VectorSpaceStateVar::Ptr> vels;

      poses.push_back(traj_states[0].getPose());
      vels.push_back(traj_states[0].getVelocity());
      poses.push_back(traj_states[2].getPose());
      vels.push_back(traj_states[2].getVelocity());

      steam::TrajErrorEval::Ptr traj_error(new steam::TrajErrorEval(poses, vels));

      auto traj_factor = steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>::Ptr(
          new steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>(
              traj_error,
              noise_model,
              loss_function));
      cost_terms->add(traj_factor);

      problem->addCostTerm(cost_terms);

      // setup solver and optimize
      std::shared_ptr<steam::GaussNewtonSolverBase> gn_solver;
      steam::DoglegGaussNewtonSolver::Params params;
      params.maxIterations = 1;
      gn_solver.reset(new steam::DoglegGaussNewtonSolver(problem.get(), params));

      try {
        gn_solver->optimize();
      } catch (steam::unsuccessful_step &e) {
        std::cout
            << "Steam has failed to optimize interpolated covariance problem! This is an ERROR."
            << std::endl;
        return Eigen::Matrix<double, 6, 6>::Identity();
      } catch (steam::decomp_failure &e) {
        // Should not occur frequently
        std::cout
            << "Steam has encountered an LL^T decomposition error while optimizing for interpolated covariance! This is an ERROR."
            << std::endl;
        return Eigen::Matrix<double, 6, 6>::Identity();
      }

      std::vector<steam::StateKey> pose_keys{statevars[1]->getKey(), statevars[3]->getKey()};
      auto Cov_a0a0_b0b0 = gn_solver->queryCovarianceBlock(pose_keys);
      lgmath::se3::Transformation T_a = getInterpPoseEval(time_a)->evaluate();
      lgmath::se3::Transformation T_b = getInterpPoseEval(time_b)->evaluate();
      lgmath::se3::Transformation T_b_a = T_b / T_a;

      std::cout << "T_b_a solved \n" << (statevars[3]->getValue() / statevars[1]->getValue()).matrix() << std::endl;
      std::cout << "T_b_a extrapolated \n" << T_b_a.matrix() << std::endl;

      auto Tadj_b_a = T_b_a.adjoint();
      auto correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
      auto Cov_ba_ba =
          Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
              Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
      std::cout << "Warning: Extrapolate relative covariance (binary) not fully tested yet." << std::endl;   // temporary
      return Cov_ba_ba;

    } else {
      // ternary bracket extrapolation case
      std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1a;
      std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_2a;
      std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1b;
      it1a->second->getPose()->getActiveStateVariables(&out_state_1a);
      it2a->second->getPose()->getActiveStateVariables(&out_state_2a);
      it1b->second->getPose()->getActiveStateVariables(&out_state_1b);
      if (out_state_1a.empty() || out_state_2a.empty() || out_state_1b.empty()) {
        // TODO: Be able to handle locked states
        throw std::runtime_error("Attempted covariance interpolation with locked states");
      }

      std::vector<steam::StateKey> keys{out_state_1a.begin()->second->getKey(),
                                        it1a->second->getVelocity()->getKey(),
                                        out_state_2a.begin()->second->getKey(),
                                        it2a->second->getVelocity()->getKey(),
                                        out_state_1b.begin()->second->getKey(),
                                        it1b->second->getVelocity()->getKey()};

      steam::BlockMatrix Cov_quad = solver_->queryCovarianceBlock(keys);  // 36 x 36

      // add our 5 state variables
      std::vector<SteamTrajVar> traj_states;
      std::vector<TransformStateVar::Ptr> statevars;

      TransformStateVar::Ptr tmp_state_1a(new TransformStateVar(it1a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_1a = TransformStateEvaluator::MakeShared(tmp_state_1a);
      VectorSpaceStateVar::Ptr tmp_vel_1a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_1a(it1a->second->getTime(), tmp_pose_1a, tmp_vel_1a);
      statevars.push_back(tmp_state_1a);
      traj_states.push_back(tmp_1a);

      TransformStateVar::Ptr tmp_state_a(new TransformStateVar(it1a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_a = TransformStateEvaluator::MakeShared(tmp_state_a);
      VectorSpaceStateVar::Ptr tmp_vel_a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_a(time_a, tmp_pose_a, tmp_vel_a);
      statevars.push_back(tmp_state_a);
      traj_states.push_back(tmp_a);

      TransformStateVar::Ptr tmp_state_2a(new TransformStateVar(it2a->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_2a = TransformStateEvaluator::MakeShared(tmp_state_2a);
      VectorSpaceStateVar::Ptr tmp_vel_2a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it2a->second->getVelocity()->getValue()));
      SteamTrajVar tmp_2a(it2a->second->getTime(), tmp_pose_2a, tmp_vel_2a);
      statevars.push_back(tmp_state_2a);
      traj_states.push_back(tmp_2a);

      TransformStateVar::Ptr tmp_state_1b(new TransformStateVar(it1b->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_1b = TransformStateEvaluator::MakeShared(tmp_state_1b);
      VectorSpaceStateVar::Ptr tmp_vel_1b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
      SteamTrajVar tmp_1b(it1b->second->getTime(), tmp_pose_1b, tmp_vel_1b);
      statevars.push_back(tmp_state_1b);
      traj_states.push_back(tmp_1b);

      TransformStateVar::Ptr tmp_state_b(new TransformStateVar(it1b->second->getPose()->evaluate()));
      TransformStateEvaluator::Ptr tmp_pose_b = TransformStateEvaluator::MakeShared(tmp_state_b);
      VectorSpaceStateVar::Ptr tmp_vel_b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
      SteamTrajVar tmp_b(time_b, tmp_pose_b, tmp_vel_b);
      statevars.push_back(tmp_state_b);
      traj_states.push_back(tmp_b);

      std::shared_ptr<steam::OptimizationProblem> problem;
      problem.reset(new steam::OptimizationProblem());

      for (auto & state : statevars) {
        problem->addStateVariable(state);
      }
      for (auto & state : traj_states) {
        problem->addStateVariable(state.getVelocity());
      }

      steam::ParallelizedCostTermCollection::Ptr cost_terms;
      cost_terms.reset(new steam::ParallelizedCostTermCollection());

      // we create two separate trajectories so we don't have smoothing terms between 2a and 1b
      steam::se3::SteamTrajInterface traj_a(Qc_inv_);
      steam::se3::SteamTrajInterface traj_b(Qc_inv_, true);
      for (int i = 0; i < 3; ++i) {
        traj_a.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
      }
      for (int i = 3; i < 5; ++i) {
        traj_b.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
      }

      traj_a.appendPriorCostTerms(cost_terms);
      traj_b.appendPriorCostTerms(cost_terms);

      // copy over posterior to
      Eigen::MatrixXd post_cov = Eigen::MatrixXd::Identity(36, 36);
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          post_cov.block<6,6>(6*i, 6*j) = Cov_quad.at(i, j);
        }
      }

      steam::BaseNoiseModel<Eigen::Dynamic>::Ptr noise_model(new steam::StaticNoiseModel<Eigen::Dynamic>(post_cov));
      steam::L2LossFunc::Ptr loss_function(new steam::L2LossFunc());

      // Set up trajectory error factor for posterior
      std::vector<se3::TransformEvaluator::Ptr> poses;
      std::vector<VectorSpaceStateVar::Ptr> vels;

      poses.push_back(traj_states[0].getPose());
      vels.push_back(traj_states[0].getVelocity());
      poses.push_back(traj_states[2].getPose());
      vels.push_back(traj_states[2].getVelocity());
      poses.push_back(traj_states[3].getPose());
      vels.push_back(traj_states[3].getVelocity());

      steam::TrajErrorEval::Ptr traj_error(new steam::TrajErrorEval(poses, vels));

      auto traj_factor = steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>::Ptr(
          new steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>(
              traj_error,
              noise_model,
              loss_function));
      cost_terms->add(traj_factor);

      problem->addCostTerm(cost_terms);

      // setup solver and optimize
      std::shared_ptr<steam::GaussNewtonSolverBase> gn_solver;
      steam::DoglegGaussNewtonSolver::Params params;
      params.maxIterations = 1;
      gn_solver.reset(new steam::DoglegGaussNewtonSolver(problem.get(), params));

      try {
        gn_solver->optimize();
      } catch (steam::unsuccessful_step &e) {
        std::cout
            << "Steam has failed to optimize interpolated covariance problem! This is an ERROR."
            << std::endl;
        return Eigen::Matrix<double, 6, 6>::Identity();
      } catch (steam::decomp_failure &e) {
        // Should not occur frequently
        std::cout
            << "Steam has encountered an LL^T decomposition error while optimizing for interpolated covariance! This is an ERROR."
            << std::endl;
        return Eigen::Matrix<double, 6, 6>::Identity();
      }

      std::vector<steam::StateKey> pose_keys{statevars[1]->getKey(), statevars[4]->getKey()};
      auto Cov_a0a0_b0b0 = gn_solver->queryCovarianceBlock(pose_keys);
      lgmath::se3::Transformation T_a = getInterpPoseEval(time_a)->evaluate();
      lgmath::se3::Transformation T_b = getInterpPoseEval(time_b)->evaluate();
      lgmath::se3::Transformation T_b_a = T_b / T_a;

      auto Tadj_b_a = T_b_a.adjoint();
      auto correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
      auto Cov_ba_ba =
          Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
              Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
      std::cout << "Warning: Extrapolate relative covariance (ternary) not fully tested yet." << std::endl;   // temporary
      return Cov_ba_ba;
    }
  }

  // Check if both query times also state times and we don't have to interpolate
  if (it1a->second->getTime() == time_a && it1b->second->getTime() == time_b) {
    // return covariance exactly (no interp)
    std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_a;
    std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_b;
    it1a->second->getPose()->getActiveStateVariables(&out_state_a);
    it1b->second->getPose()->getActiveStateVariables(&out_state_b);

    std::vector<steam::StateKey> keys{out_state_a.begin()->second->getKey(),
                                      out_state_b.begin()->second->getKey()};

    steam::BlockMatrix Cov_a0a0_b0b0 = solver_->queryCovarianceBlock(keys);

    lgmath::se3::Transformation T_b_a = it1b->second->getPose()->evaluate() / it1a->second->getPose()->evaluate();
    Eigen::Matrix<double, 6, 6> Tadj_b_a = T_b_a.adjoint();
    Eigen::Matrix<double, 6, 6> correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
    Eigen::Matrix<double, 6, 6> Cov_ba_ba = Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
            Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
    return Cov_ba_ba;
  }

  // Interpolating covariance...

  if (it1b->second->getTime() < it2a->second->getTime()){
    // Binary bracket case
    if (it1b->second->getTime() != it1a->second->getTime()) {
      throw std::runtime_error("[getRelativeCovariance]: Times out of order.");
    }

    std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1a;
    std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_2a;
    it1a->second->getPose()->getActiveStateVariables(&out_state_1a);
    it2a->second->getPose()->getActiveStateVariables(&out_state_2a);
    if (out_state_1a.empty() || out_state_2a.empty()) {
      // TODO: Be able to handle locked states
      throw std::runtime_error("Attempted covariance interpolation with locked states");
    }

    std::vector<steam::StateKey> keys{out_state_1a.begin()->second->getKey(),
                                      it1a->second->getVelocity()->getKey(),
                                      out_state_2a.begin()->second->getKey(),
                                      it2a->second->getVelocity()->getKey()};
    steam::BlockMatrix Cov_quad = solver_->queryCovarianceBlock(keys);  // 24 x 24

    // add our 4 state variables
    std::vector<SteamTrajVar> traj_states;
    std::vector<TransformStateVar::Ptr> statevars;

    TransformStateVar::Ptr tmp_state_1a(new TransformStateVar(it1a->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_1a = TransformStateEvaluator::MakeShared(tmp_state_1a);
    VectorSpaceStateVar::Ptr tmp_vel_1a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
    SteamTrajVar tmp_1a(it1a->second->getTime(), tmp_pose_1a, tmp_vel_1a);
    statevars.push_back(tmp_state_1a);
    traj_states.push_back(tmp_1a);

    TransformStateVar::Ptr tmp_state_a(new TransformStateVar(it1a->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_a = TransformStateEvaluator::MakeShared(tmp_state_a);
    VectorSpaceStateVar::Ptr tmp_vel_a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
    SteamTrajVar tmp_a(time_a, tmp_pose_a, tmp_vel_a);
    statevars.push_back(tmp_state_a);
    traj_states.push_back(tmp_a);

    TransformStateVar::Ptr tmp_state_b(new TransformStateVar(it1b->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_b = TransformStateEvaluator::MakeShared(tmp_state_b);
    VectorSpaceStateVar::Ptr tmp_vel_b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
    SteamTrajVar tmp_b(time_b, tmp_pose_b, tmp_vel_b);
    statevars.push_back(tmp_state_b);
    traj_states.push_back(tmp_b);

    TransformStateVar::Ptr tmp_state_2a(new TransformStateVar(it2a->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_2a = TransformStateEvaluator::MakeShared(tmp_state_2a);
    VectorSpaceStateVar::Ptr tmp_vel_2a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it2a->second->getVelocity()->getValue()));
    SteamTrajVar tmp_2a(it2a->second->getTime(), tmp_pose_2a, tmp_vel_2a);
    statevars.push_back(tmp_state_2a);
    traj_states.push_back(tmp_2a);

    std::shared_ptr<steam::OptimizationProblem> problem;
    problem.reset(new steam::OptimizationProblem());

    for (auto & state : statevars) {
      problem->addStateVariable(state);
    }
    for (auto & state : traj_states) {
      problem->addStateVariable(state.getVelocity());
    }

    steam::ParallelizedCostTermCollection::Ptr cost_terms;
    cost_terms.reset(new steam::ParallelizedCostTermCollection());

    // one trajectory in this case
    steam::se3::SteamTrajInterface traj_a(Qc_inv_);
    for (int i = 0; i < 4; ++i) {
      traj_a.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
    }

    traj_a.appendPriorCostTerms(cost_terms);

    // copy over posterior to
    Eigen::MatrixXd post_cov = Eigen::MatrixXd::Identity(24, 24);
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        post_cov.block<6,6>(6*i, 6*j) = Cov_quad.at(i, j);
      }
    }

    steam::BaseNoiseModel<Eigen::Dynamic>::Ptr noise_model(new steam::StaticNoiseModel<Eigen::Dynamic>(post_cov));
    steam::L2LossFunc::Ptr loss_function(new steam::L2LossFunc());

    // Set up trajectory error factor for posterior
    std::vector<se3::TransformEvaluator::Ptr> poses;
    std::vector<VectorSpaceStateVar::Ptr> vels;

    poses.push_back(traj_states[0].getPose());
    vels.push_back(traj_states[0].getVelocity());
    poses.push_back(traj_states[3].getPose());
    vels.push_back(traj_states[3].getVelocity());

    steam::TrajErrorEval::Ptr traj_error(new steam::TrajErrorEval(poses, vels));

    auto traj_factor = steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>::Ptr(
        new steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>(
            traj_error,
            noise_model,
            loss_function));
    cost_terms->add(traj_factor);

    problem->addCostTerm(cost_terms);

    // setup solver and optimize
    std::shared_ptr<steam::GaussNewtonSolverBase> gn_solver;
    steam::DoglegGaussNewtonSolver::Params params;
    params.maxIterations = 1;
    gn_solver.reset(new steam::DoglegGaussNewtonSolver(problem.get(), params));

    try {
      gn_solver->optimize();
    } catch (steam::unsuccessful_step &e) {
      std::cout
          << "Steam has failed to optimize interpolated covariance problem! This is an ERROR."
          << std::endl;
      return Eigen::Matrix<double, 6, 6>::Identity();
    } catch (steam::decomp_failure &e) {
      // Should not occur frequently
      std::cout
          << "Steam has encountered an LL^T decomposition error while optimizing for interpolated covariance! This is an ERROR."
          << std::endl;
      return Eigen::Matrix<double, 6, 6>::Identity();
    }

    std::vector<steam::StateKey> pose_keys{statevars[1]->getKey(), statevars[2]->getKey()};
    auto Cov_a0a0_b0b0 = gn_solver->queryCovarianceBlock(pose_keys);
    lgmath::se3::Transformation T_a = getInterpPoseEval(time_a)->evaluate();
    lgmath::se3::Transformation T_b = getInterpPoseEval(time_b)->evaluate();
    lgmath::se3::Transformation T_b_a = T_b / T_a;

    std::cout << "T_b_a solved \n" << (statevars[2]->getValue() / statevars[1]->getValue()).matrix() << std::endl;
    std::cout << "T_b_a interpolated \n" << T_b_a.matrix() << std::endl;

    auto Tadj_b_a = T_b_a.adjoint();
    auto correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
    auto Cov_ba_ba =
        Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
            Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
    std::cout << "Warning: Interpolate relative covariance (binary) not fully tested yet." << std::endl;   // temporary
    return Cov_ba_ba;
  } else if (it1b->second->getTime() == it2a->second->getTime()) {
    // Ternary bracket case
    std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1a;
    std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_2a;
    std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_2b;
    it1a->second->getPose()->getActiveStateVariables(&out_state_1a);
    it2a->second->getPose()->getActiveStateVariables(&out_state_2a);
    it2b->second->getPose()->getActiveStateVariables(&out_state_2b);
    if (out_state_1a.empty() || out_state_2a.empty() || out_state_2b.empty()) {
      // TODO: Be able to handle locked states
      throw std::runtime_error("Attempted covariance interpolation with locked states");
    }

    std::vector<steam::StateKey> keys{out_state_1a.begin()->second->getKey(),
                                      it1a->second->getVelocity()->getKey(),
                                      out_state_2a.begin()->second->getKey(),
                                      it2a->second->getVelocity()->getKey(),
                                      out_state_2b.begin()->second->getKey(),
                                      it2b->second->getVelocity()->getKey()};

    steam::BlockMatrix Cov_quad = solver_->queryCovarianceBlock(keys);  // should be 36 x 36

    // add our 5 state variables
    std::vector<SteamTrajVar> traj_states;
    std::vector<TransformStateVar::Ptr> statevars;

    TransformStateVar::Ptr tmp_state_1a(new TransformStateVar(it1a->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_1a = TransformStateEvaluator::MakeShared(tmp_state_1a);
    VectorSpaceStateVar::Ptr tmp_vel_1a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
    SteamTrajVar tmp_1a(it1a->second->getTime(), tmp_pose_1a, tmp_vel_1a);
    statevars.push_back(tmp_state_1a);
    traj_states.push_back(tmp_1a);

    TransformStateVar::Ptr tmp_state_a(new TransformStateVar(it1a->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_a = TransformStateEvaluator::MakeShared(tmp_state_a);
    VectorSpaceStateVar::Ptr tmp_vel_a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
    SteamTrajVar tmp_a(time_a, tmp_pose_a, tmp_vel_a);
    statevars.push_back(tmp_state_a);
    traj_states.push_back(tmp_a);

    TransformStateVar::Ptr tmp_state_2a(new TransformStateVar(it2a->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_2a = TransformStateEvaluator::MakeShared(tmp_state_2a);
    VectorSpaceStateVar::Ptr tmp_vel_2a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it2a->second->getVelocity()->getValue()));
    SteamTrajVar tmp_2a(it2a->second->getTime(), tmp_pose_2a, tmp_vel_2a);
    statevars.push_back(tmp_state_2a);
    traj_states.push_back(tmp_2a);

    TransformStateVar::Ptr tmp_state_b(new TransformStateVar(it1b->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_b = TransformStateEvaluator::MakeShared(tmp_state_b);
    VectorSpaceStateVar::Ptr tmp_vel_b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
    SteamTrajVar tmp_b(time_b, tmp_pose_b, tmp_vel_b);
    statevars.push_back(tmp_state_b);
    traj_states.push_back(tmp_b);

    TransformStateVar::Ptr tmp_state_2b(new TransformStateVar(it2b->second->getPose()->evaluate()));
    TransformStateEvaluator::Ptr tmp_pose_2b = TransformStateEvaluator::MakeShared(tmp_state_2b);
    VectorSpaceStateVar::Ptr tmp_vel_2b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it2b->second->getVelocity()->getValue()));
    SteamTrajVar tmp_2b(it2b->second->getTime(), tmp_pose_2b, tmp_vel_2b);
    statevars.push_back(tmp_state_2b);
    traj_states.push_back(tmp_2b);

    std::shared_ptr<steam::OptimizationProblem> problem;
    problem.reset(new steam::OptimizationProblem());

    for (auto & state : statevars) {
      problem->addStateVariable(state);
    }
    for (auto & state : traj_states) {
      problem->addStateVariable(state.getVelocity());
    }

    steam::ParallelizedCostTermCollection::Ptr cost_terms;
    cost_terms.reset(new steam::ParallelizedCostTermCollection());

    // one trajectory in this case
    steam::se3::SteamTrajInterface traj_a(Qc_inv_);
    for (int i = 0; i < 5; ++i) {
      traj_a.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
    }

    traj_a.appendPriorCostTerms(cost_terms);

    // copy over posterior
    Eigen::MatrixXd post_cov = Eigen::MatrixXd::Identity(36, 36);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        post_cov.block<6,6>(6*i, 6*j) = Cov_quad.at(i, j);
      }
    }

    steam::BaseNoiseModel<Eigen::Dynamic>::Ptr noise_model(new steam::StaticNoiseModel<Eigen::Dynamic>(post_cov));
    steam::L2LossFunc::Ptr loss_function(new steam::L2LossFunc());

    // Set up trajectory error factor for posterior
    std::vector<se3::TransformEvaluator::Ptr> poses;
    std::vector<VectorSpaceStateVar::Ptr> vels;

    poses.push_back(traj_states[0].getPose());
    vels.push_back(traj_states[0].getVelocity());
    poses.push_back(traj_states[2].getPose());
    vels.push_back(traj_states[2].getVelocity());
    poses.push_back(traj_states[4].getPose());
    vels.push_back(traj_states[4].getVelocity());

    steam::TrajErrorEval::Ptr traj_error(new steam::TrajErrorEval(poses, vels));

    auto traj_factor = steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>::Ptr(
        new steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>(
            traj_error,
            noise_model,
            loss_function));
    cost_terms->add(traj_factor);

    problem->addCostTerm(cost_terms);

    // setup solver and optimize
    std::shared_ptr<steam::GaussNewtonSolverBase> gn_solver;
    steam::DoglegGaussNewtonSolver::Params params;
    params.maxIterations = 1;
    gn_solver.reset(new steam::DoglegGaussNewtonSolver(problem.get(), params));

    try {
      gn_solver->optimize();
    } catch (steam::unsuccessful_step &e) {
      std::cout
          << "Steam has failed to optimize interpolated covariance problem! This is an ERROR."
          << std::endl;
      return Eigen::Matrix<double, 6, 6>::Identity();
    } catch (steam::decomp_failure &e) {
      // Should not occur frequently
      std::cout
          << "Steam has encountered an LL^T decomposition error while optimizing for interpolated covariance! This is an ERROR."
          << std::endl;
      return Eigen::Matrix<double, 6, 6>::Identity();
    }

    std::vector<steam::StateKey> pose_keys{statevars[1]->getKey(), statevars[3]->getKey()};
    auto Cov_a0a0_b0b0 = gn_solver->queryCovarianceBlock(pose_keys);
    lgmath::se3::Transformation T_a = getInterpPoseEval(time_a)->evaluate();
    lgmath::se3::Transformation T_b = getInterpPoseEval(time_b)->evaluate();
    lgmath::se3::Transformation T_b_a = T_b / T_a;

    auto Tadj_b_a = T_b_a.adjoint();
    auto correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
    auto Cov_ba_ba =
        Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
            Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
    std::cout << "Warning: Interpolate relative covariance (ternary) not fully tested yet." << std::endl;   // temporary
    return Cov_ba_ba;
  }

  // Quaternary bracket case
  std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1a;
  std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_2a;
  std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_1b;
  std::map<unsigned int, steam::StateVariableBase::Ptr> out_state_2b;
  it1a->second->getPose()->getActiveStateVariables(&out_state_1a);
  it2a->second->getPose()->getActiveStateVariables(&out_state_2a);
  it1b->second->getPose()->getActiveStateVariables(&out_state_1b);
  it2b->second->getPose()->getActiveStateVariables(&out_state_2b);
  if (out_state_1a.empty() || out_state_2a.empty() || out_state_1b.empty() || out_state_2b.empty()) {
    // TODO: Be able to handle locked states
    throw std::runtime_error("Attempted covariance interpolation with locked states");
  }

  std::vector<steam::StateKey> keys{out_state_1a.begin()->second->getKey(),
                                    it1a->second->getVelocity()->getKey(),
                                    out_state_2a.begin()->second->getKey(),
                                    it2a->second->getVelocity()->getKey(),
                                    out_state_1b.begin()->second->getKey(),
                                    it1b->second->getVelocity()->getKey(),
                                    out_state_2b.begin()->second->getKey(),
                                    it2b->second->getVelocity()->getKey()};

  steam::BlockMatrix Cov_quad = solver_->queryCovarianceBlock(keys);  // should be 48 x 48

  // add our 6 state variables
  std::vector<SteamTrajVar> traj_states;
  std::vector<TransformStateVar::Ptr> statevars;

  TransformStateVar::Ptr tmp_state_1a(new TransformStateVar(it1a->second->getPose()->evaluate()));
  TransformStateEvaluator::Ptr tmp_pose_1a = TransformStateEvaluator::MakeShared(tmp_state_1a);
  VectorSpaceStateVar::Ptr tmp_vel_1a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
  SteamTrajVar tmp_1a(it1a->second->getTime(), tmp_pose_1a, tmp_vel_1a);
  statevars.push_back(tmp_state_1a);
  traj_states.push_back(tmp_1a);

  TransformStateVar::Ptr tmp_state_a(new TransformStateVar(it1a->second->getPose()->evaluate()));
  TransformStateEvaluator::Ptr tmp_pose_a = TransformStateEvaluator::MakeShared(tmp_state_a);
  VectorSpaceStateVar::Ptr tmp_vel_a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1a->second->getVelocity()->getValue()));
  SteamTrajVar tmp_a(time_a, tmp_pose_a, tmp_vel_a);
  statevars.push_back(tmp_state_a);
  traj_states.push_back(tmp_a);

  TransformStateVar::Ptr tmp_state_2a(new TransformStateVar(it2a->second->getPose()->evaluate()));
  TransformStateEvaluator::Ptr tmp_pose_2a = TransformStateEvaluator::MakeShared(tmp_state_2a);
  VectorSpaceStateVar::Ptr tmp_vel_2a = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it2a->second->getVelocity()->getValue()));
  SteamTrajVar tmp_2a(it2a->second->getTime(), tmp_pose_2a, tmp_vel_2a);
  statevars.push_back(tmp_state_2a);
  traj_states.push_back(tmp_2a);

  TransformStateVar::Ptr tmp_state_1b(new TransformStateVar(it1b->second->getPose()->evaluate()));
  TransformStateEvaluator::Ptr tmp_pose_1b = TransformStateEvaluator::MakeShared(tmp_state_1b);
  VectorSpaceStateVar::Ptr tmp_vel_1b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
  SteamTrajVar tmp_1b(it1b->second->getTime(), tmp_pose_1b, tmp_vel_1b);
  statevars.push_back(tmp_state_1b);
  traj_states.push_back(tmp_1b);

  TransformStateVar::Ptr tmp_state_b(new TransformStateVar(it1b->second->getPose()->evaluate()));
  TransformStateEvaluator::Ptr tmp_pose_b = TransformStateEvaluator::MakeShared(tmp_state_b);
  VectorSpaceStateVar::Ptr tmp_vel_b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it1b->second->getVelocity()->getValue()));
  SteamTrajVar tmp_b(time_b, tmp_pose_b, tmp_vel_b);
  statevars.push_back(tmp_state_b);
  traj_states.push_back(tmp_b);

  TransformStateVar::Ptr tmp_state_2b(new TransformStateVar(it2b->second->getPose()->evaluate()));
  TransformStateEvaluator::Ptr tmp_pose_2b = TransformStateEvaluator::MakeShared(tmp_state_2b);
  VectorSpaceStateVar::Ptr tmp_vel_2b = VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(it2b->second->getVelocity()->getValue()));
  SteamTrajVar tmp_2b(it2b->second->getTime(), tmp_pose_2b, tmp_vel_2b);
  statevars.push_back(tmp_state_2b);
  traj_states.push_back(tmp_2b);

  std::shared_ptr<steam::OptimizationProblem> problem;
  problem.reset(new steam::OptimizationProblem());

  for (auto & state : statevars) {
    problem->addStateVariable(state);
  }
  for (auto & state : traj_states) {
    problem->addStateVariable(state.getVelocity());
  }

  steam::ParallelizedCostTermCollection::Ptr cost_terms;
  cost_terms.reset(new steam::ParallelizedCostTermCollection());

  // we create two separate trajectories so we don't have smoothing terms between 2a and 1b
  steam::se3::SteamTrajInterface traj_a(Qc_inv_);
  steam::se3::SteamTrajInterface traj_b(Qc_inv_);
  for (int i = 0; i < 3; ++i) {
    traj_a.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
  }
  for (int i = 3; i < 6; ++i) {
    traj_b.add(traj_states[i].getTime(), traj_states[i].getPose(), traj_states[i].getVelocity());
  }

  traj_a.appendPriorCostTerms(cost_terms);
  traj_b.appendPriorCostTerms(cost_terms);

  // copy over posterior to
  Eigen::MatrixXd post_cov = Eigen::MatrixXd::Identity(48, 48);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      post_cov.block<6,6>(6*i, 6*j) = Cov_quad.at(i, j);
    }
  }

  steam::BaseNoiseModel<Eigen::Dynamic>::Ptr noise_model(new steam::StaticNoiseModel<Eigen::Dynamic>(post_cov));
  steam::L2LossFunc::Ptr loss_function(new steam::L2LossFunc());

  // Set up trajectory error factor for posterior
  std::vector<se3::TransformEvaluator::Ptr> poses;
  std::vector<VectorSpaceStateVar::Ptr> vels;

  poses.push_back(traj_states[0].getPose());
  vels.push_back(traj_states[0].getVelocity());
  poses.push_back(traj_states[2].getPose());
  vels.push_back(traj_states[2].getVelocity());
  poses.push_back(traj_states[3].getPose());
  vels.push_back(traj_states[3].getVelocity());
  poses.push_back(traj_states[5].getPose());
  vels.push_back(traj_states[5].getVelocity());

  steam::TrajErrorEval::Ptr traj_error(new steam::TrajErrorEval(poses, vels));

  auto traj_factor = steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>::Ptr(
      new steam::WeightedLeastSqCostTerm<Eigen::Dynamic, 6>(
          traj_error,
          noise_model,
          loss_function));
  cost_terms->add(traj_factor);

  problem->addCostTerm(cost_terms);

  // setup solver and optimize
  std::shared_ptr<steam::GaussNewtonSolverBase> gn_solver;
  steam::DoglegGaussNewtonSolver::Params params;
  params.maxIterations = 1;
  gn_solver.reset(new steam::DoglegGaussNewtonSolver(problem.get(), params));

  try {
    gn_solver->optimize();
  } catch (steam::unsuccessful_step &e) {
      std::cout
          << "Steam has failed to optimize interpolated covariance problem! This is an ERROR."
          << std::endl;
      return Eigen::Matrix<double, 6, 6>::Identity();
  } catch (steam::decomp_failure &e) {
    // Should not occur frequently
    std::cout
        << "Steam has encountered an LL^T decomposition error while optimizing for interpolated covariance! This is an ERROR."
        << std::endl;
    return Eigen::Matrix<double, 6, 6>::Identity();
  }

  std::vector<steam::StateKey> pose_keys{statevars[1]->getKey(), statevars[4]->getKey()};
  auto Cov_a0a0_b0b0 = gn_solver->queryCovarianceBlock(pose_keys);
  lgmath::se3::Transformation T_a = getInterpPoseEval(time_a)->evaluate();
  lgmath::se3::Transformation T_b = getInterpPoseEval(time_b)->evaluate();
  lgmath::se3::Transformation T_b_a = T_b / T_a;

  auto Tadj_b_a = T_b_a.adjoint();
  auto correlation = Tadj_b_a * Cov_a0a0_b0b0.at(0, 1);
  auto Cov_ba_ba =
      Cov_a0a0_b0b0.at(1, 1) - correlation - correlation.transpose() +
          Tadj_b_a * Cov_a0a0_b0b0.at(0, 0) * Tadj_b_a.transpose();
  std::cout << "Warning: Interpolate relative covariance (quaternary) not fully tested yet." << std::endl;   // temporary
  return Cov_ba_ba;
}


} // se3
} // steam
