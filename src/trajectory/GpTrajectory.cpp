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
void GpTrajectory::add(const steam::Time& time, const se3::TransformEvaluator::Ptr& T_k0,
                       const VectorSpaceStateVar::Ptr& varpi) {

  // Check velocity input
  if (varpi->getPerturbDim() != 6) {
    throw std::invalid_argument("invalid velocity size");
  }

  // Make knot
  Knot::Ptr newEntry(new Knot());
  newEntry->time = time;
  newEntry->T_k_root = T_k0;
  newEntry->varpi = varpi;

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
    //return TransformStateEvaluator::MakeShared(it1->second->T_k0);
    return it1->second->T_k_root;
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
/// \brief Get cost terms associated with the prior for unlocked parts of the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void GpTrajectory::getBinaryPriorFactors(const CostTermCollectionX::Ptr& binary) const {

  // If empty, return none
  if (knotMap_.empty()) {
    return;
  }

  // All prior factors will use an L2 loss function
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Initialize first iterator
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it1 = knotMap_.begin();
  if (it1 == knotMap_.end()) {
    throw std::runtime_error("No knots...");
  }

  // Iterate through all states.. if any are unlocked, supply a prior term
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it2 = it1; ++it2;
  for (; it2 != knotMap_.end(); ++it1, ++it2) {

    // Get knots
    const Knot::ConstPtr& knot1 = it1->second;
    const Knot::ConstPtr& knot2 = it2->second;

    // Check if any of the variables are unlocked
    if(knot1->T_k_root->isActive()  || !knot1->varpi->isLocked() ||
       knot2->T_k_root->isActive()  || !knot2->varpi->isLocked() ) {

      // Generate 12 x 12 information matrix for GP prior factor
      Eigen::Matrix<double,12,12> Qi_inv;
      double one_over_dt = 1.0/(knot2->time - knot1->time).seconds();
      double one_over_dt2 = one_over_dt*one_over_dt;
      double one_over_dt3 = one_over_dt2*one_over_dt;
      Qi_inv.block<6,6>(0,0) = 12.0 * one_over_dt3 * Qc_inv_;
      Qi_inv.block<6,6>(6,0) = Qi_inv.block<6,6>(0,6) = -6.0 * one_over_dt2 * Qc_inv_;
      Qi_inv.block<6,6>(6,6) =  4.0 * one_over_dt  * Qc_inv_;
      steam::NoiseModelX::Ptr sharedGPNoiseModel(new steam::NoiseModelX(Qi_inv, steam::NoiseModelX::INFORMATION));

      // Create cost term
      steam::se3::GpTrajectoryPrior::Ptr errorfunc(new steam::se3::GpTrajectoryPrior(knot1, knot2));
      steam::CostTermX::Ptr cost(new steam::CostTermX(errorfunc, sharedGPNoiseModel, sharedLossFunc));
      binary->add(cost);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get active state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
void GpTrajectory::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  // Iterate over trajectory
  std::map<boost::int64_t, Knot::Ptr>::const_iterator it;
  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {

    // Append active states in transform evaluator
    it->second->T_k_root->getActiveStateVariables(outStates);

    // Check if velocity is locked
    if (!it->second->varpi->isLocked()) {
      (*outStates)[it->second->varpi->getKey().getID()] = it->second->varpi;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get unlocked state variables in the trajectory
//////////////////////////////////////////////////////////////////////////////////////////////
//std::vector<steam::StateVariableBase::Ptr> GpTrajectory::getActiveStateVariables() const {

//  std::vector<steam::StateVariableBase::Ptr> result;

//  // Iterate over trajectory
//  std::map<boost::int64_t, Knot::Ptr>::const_iterator it;
//  for (it = knotMap_.begin(); it != knotMap_.end(); ++it) {

//    // Check if transform is locked
//    if (!it->second->T_k0->isLocked()) {
//      result.push_back(it->second->T_k0);
//    }

//    // Check if velocity is locked
//    if (!it->second->varpi->isLocked()) {
//      result.push_back(it->second->varpi);
//    }
//  }

//  return result;
//}

} // se3
} // steam
