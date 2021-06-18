//////////////////////////////////////////////////////////////////////////////////////////////
/// \file PointToPointErrorEval2.cpp
///
/// \author Yuchen Wu, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/samples/PointToPointErrorEval2.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convenience constructor -
//////////////////////////////////////////////////////////////////////////////////////////////
PointToPointErrorEval2::PointToPointErrorEval2(
    const se3::TransformEvaluator::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query)
    : T_rq_(T_rq) {
  D_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  reference_.block<3, 1>(0, 0) = reference;
  query_.block<3, 1>(0, 0) = query;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool PointToPointErrorEval2::isActive() const { return T_rq_->isActive(); }

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 3-d measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 3, 1> PointToPointErrorEval2::evaluate() const {
  return D_ * (reference_ - T_rq_->evaluate() * query_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 3-d measurement error and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 3, 1> PointToPointErrorEval2::evaluate(
    const Eigen::Matrix<double, 3, 3> &lhs,
    std::vector<Jacobian<3, 6>> *jacs) const {
  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument(
        "Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation tree
  EvalTreeHandle<lgmath::se3::Transformation> blkAutoEvalPosOfTransformDiff =
      T_rq_->getBlockAutomaticEvaluation();

  // Get evaluation from tree
  const lgmath::se3::Transformation T_rq =
      blkAutoEvalPosOfTransformDiff.getValue();
  Eigen::Matrix<double, 3, 1> error = D_ * (reference_ - T_rq * query_);

  // Get Jacobians
  const Eigen::Matrix<double, 3, 1> Tq = (T_rq * query_).block<3, 1>(0, 0);
  const Eigen::Matrix<double, 3, 6> new_lhs =
      -lhs * D_ * lgmath::se3::point2fs(Tq);
  T_rq_->appendBlockAutomaticJacobians(
      new_lhs, blkAutoEvalPosOfTransformDiff.getRoot(), jacs);

  // Return evaluation
  return error;
}

}  // namespace steam