//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvaluators.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/GpTrajectory.hpp>

#include <lgmath.hpp>

#include <steam/trajectory/GpTrajectoryEval.hpp>
#include <steam/trajectory/GpTrajectoryPrior.hpp>
#include <steam/evaluator/common/VectorSpaceErrorEval.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
GpTrajectory::GpTrajectory(const Eigen::Matrix<double,6,6>& Qc_inv) : Qc_inv_(Qc_inv) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot
//////////////////////////////////////////////////////////////////////////////////////////////
void GpTrajectory::add(const steam::Time& time, const lgmath::se3::Transformation& T_k0, const Eigen::Matrix<double,6,1>& varpi) {

  // Make knot
  Knot::Ptr newEntry(new Knot());
  newEntry->time = time;
  newEntry->T_k0 = se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_k0));
  newEntry->varpi = VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(varpi));

  if (knotMap_.empty()) {

    // Lock first pose
    newEntry->T_k0->setLock(true);
  } else {

    // Check that time is `advancing'
    if (time.nanosecs() <= knotMap_.rbegin()->first) {
      throw std::runtime_error("Tried to add a knot in the middle of the curve");
    }
  }

  // Insert in map
  knotMap_.insert(knotMap_.end(), std::pair<boost::int64_t, Knot::Ptr>(time.nanosecs(), newEntry));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot. Initialize varpi using constant velocity between this and last pose
//////////////////////////////////////////////////////////////////////////////////////////////
void GpTrajectory::add(const steam::Time& time, const lgmath::se3::Transformation& T_k0) {

  Knot::Ptr newEntry(new Knot());
  newEntry->time = time;
  newEntry->T_k0 = se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_k0));

  if (knotMap_.empty()) {

    // Lock first pose
    newEntry->T_k0->setLock(true);

    // Initialize velocity at zero.. we have no better guess
    newEntry->varpi = VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(Eigen::Matrix<double,6,1>::Zero()));

  } else {

    // Get iterator to last element
    std::map<boost::int64_t, Knot::Ptr>::reverse_iterator rit = knotMap_.rbegin();

    // Check that time is `advancing'
    if (time.nanosecs() <= rit->first) {
      throw std::runtime_error("Tried to add a knot in the middle of the curve");
    }

    // Estimate velocity
    double deltaTime = (time - rit->second->time).seconds();
    Eigen::Matrix<double,6,1> varpi = (1.0/deltaTime) * (T_k0/rit->second->T_k0->getValue()).vec();
    newEntry->varpi = VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(varpi));
  }

  // Insert in map
  knotMap_.insert(knotMap_.end(), std::pair<boost::int64_t, Knot::Ptr>(time.nanosecs(), newEntry));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a new knot. Initialize varpi using constant velocity between this and last pose
//////////////////////////////////////////////////////////////////////////////////////////////
void GpTrajectory::add(const steam::Time& time) {

  Knot::Ptr newEntry(new Knot());
  newEntry->time = time;

  if (knotMap_.empty()) {

    // Init pose to identity
    newEntry->T_k0 = se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(lgmath::se3::Transformation()));

    // Lock first pose
    newEntry->T_k0->setLock(true);

    // Initialize velocity at zero.. we have no better guess
    newEntry->varpi = VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(Eigen::Matrix<double,6,1>::Zero()));

  } else {

    // Get iterator to last element
    std::map<boost::int64_t, Knot::Ptr>::reverse_iterator rit = knotMap_.rbegin();

    // Check that time is `advancing'
    if (time.nanosecs() <= rit->first) {
      throw std::runtime_error("Tried to add a knot in the middle of the curve");
    }

    // Extrapolate pose
    Eigen::Matrix<double,6,1> xi = (time - rit->second->time).seconds()*rit->second->varpi->getValue();
    lgmath::se3::Transformation T_k0 = lgmath::se3::Transformation(xi)*rit->second->T_k0->getValue();
    newEntry->T_k0 = se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_k0));

    // Estimate velocity
    newEntry->varpi = VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(rit->second->varpi->getValue()));

//    lgmath::se3::Transformation T_k0 = rit->second->T_k0->getValue();
//    newEntry->T_k0 = se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_k0));
//    newEntry->varpi = VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(Eigen::Matrix<double,6,1>::Zero()));
  }

  // Insert in map
  knotMap_.insert(knotMap_.end(), std::pair<boost::int64_t, Knot::Ptr>(time.nanosecs(), newEntry));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
TransformEvaluator::ConstPtr GpTrajectory::getEvaluator(const steam::Time& time) const {

  // Get first iterator
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it1 = knotMap_.lower_bound(time.nanosecs());
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time.");
  }

  // Check if we requested time exactly
  if (it1->second->time == time) {

    // return state variable exactly (no interp)
    return TransformStateEvaluator::MakeShared(it1->second->T_k0);
  }

  // Get `earlier' iterator
  if (it1 == knotMap_.begin()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it2 = it1; --it1;
  if (time <= it1->second->time || time >= it2->second->time) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // Create interpolated evaluator
  return GpTrajectoryEval::MakeShared(time, it1->second, it2->second);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Locks state variables before the provided time. Useful in sliding window filters.
//////////////////////////////////////////////////////////////////////////////////////////////
void GpTrajectory::lockBefore(const steam::Time& time) {

  std::map<boost::int64_t, Knot::Ptr>::reverse_iterator rit;
  for (rit = knotMap_.rbegin(); rit != knotMap_.rend(); ++rit) {
    if (rit->second->time < time) {
      rit->second->T_k0->setLock(true);
      rit->second->varpi->setLock(true);
      // todo could check for an early stop.. ?
    }
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get unlocked state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
std::vector<steam::StateVariableBase::Ptr> GpTrajectory::getActiveStateVariables() const {

  std::vector<steam::StateVariableBase::Ptr> result;

  // Iterate over trajectory
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {

    // Check if transform is locked
    if (!it->second->T_k0->isLocked()) {
      result.push_back(it->second->T_k0);
    }

    // Check if velocity is locked
    if (!it->second->varpi->isLocked()) {
      result.push_back(it->second->varpi);
    }
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get cost terms associated with the prior for unlocked parts of the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
std::vector<steam::CostTerm::ConstPtr> GpTrajectory::getPriorCostTerms() const {

  std::vector<steam::CostTerm::ConstPtr> costTerms;

  // If empty, return none
  if (knotMap_.empty()) {
    return costTerms;
  }

  // All prior factors will use an L2 loss function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Add initial prior terms if variables are not locked
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it1 = knotMap_.begin();
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("Requested trajectory evaluator at an invalid time. This exception "
                             "should not trigger... report to a STEAM contributor.");
  }

  // If initial pose is unlocked, add a prior
  if (!it1->second->T_k0->isLocked()) {
    throw std::logic_error("Behaviour has changed and the initial pose is not locked by default..."
                           "an initial prior term needs to be added.");
  }

  // If initial velocity is unlocked, add a prior
  //  **Note: in general, this prior may not be required if there is adequate measurements, but
  //          is needed to make a standalone prior that is well conditioned.
  if (!it1->second->varpi->isLocked()) {

    // Setup noise for initial velocity (very uncertain)
    steam::NoiseModel::Ptr initialVelocityNoiseModel(new steam::NoiseModel(10000.0*Eigen::MatrixXd::Identity(6,6)));

    // Setup zero measurement
    Eigen::VectorXd meas = Eigen::Matrix<double,6,1>::Zero();

    // Setup unary error and cost term
    steam::VectorSpaceErrorEval::Ptr errorfunc(new steam::VectorSpaceErrorEval(meas, it1->second->varpi));
    steam::CostTerm::Ptr cost(new steam::CostTerm(errorfunc, initialVelocityNoiseModel, sharedLossFunc));
    costTerms.push_back(cost);
  }

  // Iterate through all states.. if any are unlocked, supply a prior term
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it2 = it1; ++it2;
  for (; it2 != knotMap_.end(); ++it1, ++it2) {

    // Get knots
    const Knot::ConstPtr& knot1 = it1->second;
    const Knot::ConstPtr& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if(!knot1->T_k0->isLocked()  || !knot1->varpi->isLocked() ||
       !knot2->T_k0->isLocked()  || !knot2->varpi->isLocked() ) {

      // Generate 12 x 12 covariance/information matrix for GP prior factor
      Eigen::Matrix<double,12,12> Qi_inv;
      double one_over_dt = 1.0/(knot2->time - knot1->time).seconds();
      double one_over_dt2 = one_over_dt*one_over_dt;
      double one_over_dt3 = one_over_dt2*one_over_dt;
      Qi_inv.block<6,6>(0,0) = 12.0 * one_over_dt3 * Qc_inv_;
      Qi_inv.block<6,6>(6,0) = Qi_inv.block<6,6>(0,6) = -6.0 * one_over_dt2 * Qc_inv_;
      Qi_inv.block<6,6>(6,6) =  4.0 * one_over_dt  * Qc_inv_;
      steam::NoiseModel::Ptr sharedGPNoiseModel(new steam::NoiseModel(Qi_inv, steam::NoiseModel::INFORMATION));

      // Create cost term
      steam::se3::GpTrajectoryPrior::Ptr errorfunc(new steam::se3::GpTrajectoryPrior(knot1, knot2));
      steam::CostTerm::Ptr cost(new steam::CostTerm(errorfunc, sharedGPNoiseModel, sharedLossFunc));
      costTerms.push_back(cost);
    }
  }

  // Return
  return costTerms;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get all of the transformation state variables
//////////////////////////////////////////////////////////////////////////////////////////////
std::vector<se3::TransformStateVar::Ptr> GpTrajectory::getTransformStateVariables() const {

  std::vector<se3::TransformStateVar::Ptr> result;

  // Iterate over trajectory
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {
    result.push_back(it->second->T_k0);
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get all of the velocity state variables
//////////////////////////////////////////////////////////////////////////////////////////////
std::vector<VectorSpaceStateVar::Ptr> GpTrajectory::getVelocityStateVariables() const {

  std::vector<VectorSpaceStateVar::Ptr> result;

  // Iterate over trajectory
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {
    result.push_back(it->second->varpi);
  }

  return result;
}

} // se3
} // steam
