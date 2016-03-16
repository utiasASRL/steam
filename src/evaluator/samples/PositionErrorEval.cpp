//////////////////////////////////////////////////////////////////////////////////////////////
/// \file PositionErrorEval.cpp
///
/// \author Kai van Es, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/samples/PositionErrorEval.hpp>

//#include <steam/evaluator/blockauto/transform/TransformStateEvaluator.hpp>
//#include <steam/evaluator/blockauto/transform/FixedTransformEvaluator.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor - error is difference between 'T' and zero
//////////////////////////////////////////////////////////////////////////////////////////////
PositionErrorEval::PositionErrorEval(const se3::TransformEvaluator::ConstPtr &T) {
  errorEvaluator_.reset(new se3::PositionEvaluator(T));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convenience constructor - error between meas_T_21 and T_21
//////////////////////////////////////////////////////////////////////////////////////////////
PositionErrorEval::PositionErrorEval(const lgmath::se3::Transformation &meas_T_21,
                                     const se3::TransformEvaluator::ConstPtr &T_21) {

  // Construct the evaluator using the convenient transform evaluators
  se3::FixedTransformEvaluator::ConstPtr meas = se3::FixedTransformEvaluator::MakeShared(meas_T_21);
  errorEvaluator_.reset(new se3::PositionEvaluator(se3::compose(meas, se3::inverse(T_21))));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convenience constructor - linear error between meas_r_21_in1 and T_21
//////////////////////////////////////////////////////////////////////////////////////////////
PositionErrorEval::PositionErrorEval(const Eigen::Vector3d &meas_r_21_in1,
                                     const se3::TransformEvaluator::ConstPtr &T_21) {

  // Build a transform that expresses a frame at point meas_r_21.
  lgmath::se3::Transformation meas_T_21(Eigen::Matrix3d::Identity(), meas_r_21_in1);

  // Construct the evaluator using the convenient transform evaluators
  se3::FixedTransformEvaluator::ConstPtr meas = se3::FixedTransformEvaluator::MakeShared(meas_T_21);
  errorEvaluator_.reset(new se3::PositionEvaluator(se3::compose(meas, se3::inverse(T_21))));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convenience constructor - error between meas_T_21 and T_20*inv(T_10)
//////////////////////////////////////////////////////////////////////////////////////////////
PositionErrorEval::PositionErrorEval(const lgmath::se3::Transformation &meas_T_21,
                                     const se3::TransformStateVar::Ptr &T_20,
                                     const se3::TransformStateVar::Ptr &T_10) {

  // Construct the evaluator using the convenient transform evaluators
  se3::FixedTransformEvaluator::ConstPtr meas = se3::FixedTransformEvaluator::MakeShared(meas_T_21);
  se3::TransformStateEvaluator::ConstPtr t10 = se3::TransformStateEvaluator::MakeShared(T_10);
  se3::TransformStateEvaluator::ConstPtr t20 = se3::TransformStateEvaluator::MakeShared(T_20);
  errorEvaluator_.reset(new se3::PositionEvaluator(se3::compose(se3::compose(meas, t10), se3::inverse(t20))));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convenience constructor - error between meas_r_21_in1 and T_20*inv(T_10)
//////////////////////////////////////////////////////////////////////////////////////////////
PositionErrorEval::PositionErrorEval(const Eigen::Vector3d &meas_r_21_in1,
                                     const se3::TransformStateVar::Ptr &T_20,
                                     const se3::TransformStateVar::Ptr &T_10) {

  // Build a transform that expresses a frame at point meas_r_21.
  lgmath::se3::Transformation meas_T_21(Eigen::Matrix3d::Identity(), meas_r_21_in1);

  // Construct the evaluator using the convenient transform evaluators
  se3::FixedTransformEvaluator::ConstPtr meas = se3::FixedTransformEvaluator::MakeShared(meas_T_21);
  se3::TransformStateEvaluator::ConstPtr t10 = se3::TransformStateEvaluator::MakeShared(T_10);
  se3::TransformStateEvaluator::ConstPtr t20 = se3::TransformStateEvaluator::MakeShared(T_20);
  errorEvaluator_.reset(new se3::PositionEvaluator(se3::compose(se3::compose(meas, t10), se3::inverse(t20))));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool PositionErrorEval::isActive() const {
  return errorEvaluator_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 3-d measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 3, 1> PositionErrorEval::evaluate() const {
  return errorEvaluator_->evaluate();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 3-d measurement error and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double, 3, 1> PositionErrorEval::evaluate(const Eigen::Matrix<double, 3, 3> &lhs,
                                                        std::vector<Jacobian<3, 6> > *jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation tree
  EvalTreeHandle<Eigen::Matrix<double, 3, 1> > blkAutoEvalPosOfTransformDiff =
      errorEvaluator_->getBlockAutomaticEvaluation();

  // Get evaluation from tree
  Eigen::Matrix<double, 3, 1> error = blkAutoEvalPosOfTransformDiff.getValue();

  // Get Jacobians
  errorEvaluator_->appendBlockAutomaticJacobians(lhs,
                                                 blkAutoEvalPosOfTransformDiff.getRoot(), jacs);

  // Return evaluation
  return error;
}

} // steam
