#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace traj {
namespace const_vel {

/**
 * \brief This class wraps a pose and velocity evaluator to act as a
 * discrete-time trajectory state variable for continuous-time trajectory
 * estimation.
 */
class Variable {
 public:
  /// Shared pointer typedefs for readability
  using Ptr = std::shared_ptr<Variable>;
  using ConstPtr = std::shared_ptr<const Variable>;

  using PoseType = lgmath::se3::Transformation;
  using VelocityType = Eigen::Matrix<double, 6, 1>;
  // using CovType = Eigen::Matrix<double, 12, 12>;
  using CovType = Eigen::MatrixXd;

  static Ptr MakeShared(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
                        const Evaluable<VelocityType>::Ptr& w_0k_ink);
  // static Ptr MakeShared(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
  //                       const Evaluable<VelocityType>::Ptr& w_0k_ink,
  //                       const CovType& cov);

  /** \brief Constructor */
  Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
           const Evaluable<VelocityType>::Ptr& w_0k_ink);
  // Variable(const Time& time, const Evaluable<PoseType>::Ptr& T_k0,
  //          const Evaluable<VelocityType>::Ptr& w_0k_ink, const CovType& cov);

  /** \brief Constructor \todo should not be a base class */
  virtual ~Variable() = default;

  /** \brief Get timestamp */
  const Time& getTime() const;

  /** \brief Get pose evaluator */
  const Evaluable<PoseType>::Ptr& getPose() const;

  /** \brief Get velocity state variable */
  const Evaluable<VelocityType>::Ptr& getVelocity() const;

  /** \brief Get keys for pose and velocity if they are active */
  std::vector<StateKey> getActiveKeys() const;

  /** \brief Get covariance matrix */
  const CovType& getCovariance() const;

  /** \brief Get cross-covariance with previous knot (1 row above) */
  const CovType& getCovPrev() const;

  /** \brief Get cross-covariance with next knot (1 row below) */
  const CovType& getCovNext() const;

  /** \brief Returns true if covariance is set */
  bool covarianceSet() const;

  /** \brief Returns true if cross-covariance with previous knot is set */
  bool covPrevSet() const;

  /** \brief Returns true if cross-covariance with next knot is set */
  bool covNextSet() const;

  /** \brief Set covariance matrix */
  void setCovariance(const CovType& cov);

  /** \brief Set cross-covariance with previous knot (1 row above) */
  void setCovPrev(const CovType& cov);

  /** \brief Set cross-covariance with next knot (1 row below) */
  void setCovNext(const CovType& cov);

  /** \brief Set bool set variables for all covariances to be false */
  void resetCovariance();

 private:
  /** \brief Timestamp of trajectory variable */
  Time time_;

  /** \brief Pose evaluator */
  const Evaluable<PoseType>::Ptr T_k0_;

  /** \brief Generalized 6D velocity state variable */
  const Evaluable<VelocityType>::Ptr w_0k_ink_;

  /** \brief Covariance */
  CovType cov_;
  bool cov_set_ = false;

  /** \brief State keys for adjacent knots along trajectory */
  std::vector<steam::StateKey> prev_keys_;
  std::vector<steam::StateKey> next_keys_;

  /** \brief Cross covariances for adjacent knots along trajectory */
  CovType prev_cov_;
  CovType next_cov_;
  bool prev_cov_set_ = false;
  bool next_cov_set_ = false;
};

}  // namespace const_vel
}  // namespace traj
}  // namespace steam