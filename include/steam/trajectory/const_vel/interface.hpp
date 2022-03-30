#pragma once

#include <Eigen/Core>

#include "steam/problem/OptimizationProblem.hpp"
#include "steam/problem/WeightedLeastSqCostTerm.hpp"
#include "steam/trajectory/const_vel/variable.hpp"
#include "steam/trajectory/interface.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_vel {

/**
 * \brief The trajectory class wraps a set of state variables to provide an
 * interface that allows for continuous-time pose interpolation.
 */
class Interface : public traj::Interface {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<Interface>;
  using ConstPtr = std::shared_ptr<const Interface>;

  using PoseType = lgmath::se3::Transformation;
  using VelocityType = Eigen::Matrix<double, 6, 1>;
  using CovType = Eigen::Matrix<double, 12, 12>;

  /**
   * \brief Constructor
   *        Note that the weighting matrix, Qc, should be provided if prior
   *        factors are needed for estimation. Without Qc the interpolation
   *        methods can be used safely.
   */
  Interface(const bool allowExtrapolation = false);
  Interface(const Eigen::Matrix<double, 6, 6>& Qc_inv,
            const bool allowExtrapolation = false);

  /** \brief Add a new knot */
  void add(const Variable::Ptr& knot);
  void add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
           const Evaluable<VelocityType>::Ptr& w_0k_ink);
  void add(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
           const Evaluable<VelocityType>::Ptr& w_0k_ink, const CovType& cov);

  /** \brief Get transform evaluator */
  Evaluable<PoseType>::ConstPtr getPoseInterpolator(const Time& time) const;

  /** \brief Get velocity evaluator */
  Evaluable<VelocityType>::ConstPtr getVelocityInterpolator(
      const Time& time) const;

  /**
   * \brief Add a unary pose prior factor at a knot time. Note that only a
   * single pose prior should exist on a trajectory, adding a second will
   * overwrite the first.
   */
  void addPosePrior(const Time& time, const PoseType& T_k0,
                    const Eigen::Matrix<double, 6, 6>& cov);

  /**
   * \brief Add a unary velocity prior factor at a knot time. Note that only a
   * single velocity prior should exist on a trajectory, adding a second will
   * overwrite the first.
   */
  void addVelocityPrior(const Time& time, const VelocityType& w_0k_ink,
                        const Eigen::Matrix<double, 6, 6>& cov);

  /**
   * \brief Get binary cost terms associated with the prior for active parts of
   * the trajectory
   */
  void addPriorCostTerms(OptimizationProblem& problem) const override;

 protected:
  /** \brief Ordered map of knots */
  Eigen::Matrix<double, 6, 6> Qc_inv_;

  /** \brief Allow for extrapolation */
  const bool allowExtrapolation_;

  /** \brief Pose prior */
  WeightedLeastSqCostTerm<6>::Ptr posePriorFactor_;

  /** \brief Velocity prior */
  WeightedLeastSqCostTerm<6>::Ptr velocityPriorFactor_;

  /** \brief Ordered map of knots */
  std::map<Time, Variable::Ptr> knotMap_;
};

}  // namespace const_vel
}  // namespace traj
}  // namespace steam