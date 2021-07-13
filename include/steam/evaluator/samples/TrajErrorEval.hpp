#pragma once

#include <steam.hpp>

namespace steam {

/** \brief Applies a posterior matrix on a trajectory for use in a new optimization problem */
class TrajErrorEval : public ErrorEvaluator<Eigen::Dynamic, 6>::type {
 public:
  /** \brief Constructor */
  TrajErrorEval(const std::vector<se3::TransformEvaluator::Ptr>& poses,
                const std::vector<VectorSpaceStateVar::Ptr>& vels);

  /** \brief Returns whether or not an evaluator contains unlocked state variables */
  virtual bool isActive() const;

  /** \brief Evaluate the 12*n_-d measurement error */
  virtual Eigen::Matrix<double, Eigen::Dynamic, 1> evaluate() const;

  /** \brief Evaluate the 12*n_-d measurement error and Jacobians */
  virtual Eigen::Matrix<double, Eigen::Dynamic, 1> evaluate(
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &lhs,
      std::vector<Jacobian<Eigen::Dynamic, 6> > *jacs) const;

 private:

  std::vector<se3::LogMapEvaluator::Ptr> pose_evals_;
  std::vector<VectorSpaceErrorEval<6, 6>::Ptr> vel_evals_;
  std::vector<VectorSpaceStateVar::Ptr> vels_;

  /** \brief size of pose_evals_, vel_evals_. State vector will be 12*n_ */
  uint n_;

};
} // steam