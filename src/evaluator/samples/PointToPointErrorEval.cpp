//////////////////////////////////////////////////////////////////////////////////////////////
/// \file PointToPointErrorEval.cpp
///
/// \author Francois Pomerleau, ASRL
/// \brief This evaluator was develop in the context of ICP (Iterative Closest Point) 
///        implementation.
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/samples/PointToPointErrorEval.hpp>

namespace steam {


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
PointToPointErrorEval::PointToPointErrorEval(const Eigen::Vector4d& ref_a,
																						 const se3::TransformEvaluator::ConstPtr& T_a_world,
																						 const Eigen::Vector4d& read_b,
																						 const se3::TransformEvaluator::ConstPtr& T_b_world
																						 )
  : ref_a_(ref_a), 
	  T_ab_(se3::ComposeInverseTransformEvaluator::MakeShared(T_a_world, T_b_world)),
		read_b_(read_b)	{
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool PointToPointErrorEval::isActive() const {
  return T_ab_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d PointToPointErrorEval::evaluate() const {

  // Return error (between measurement and point estimate projected in camera frame)
  return ref_a_ - (T_ab_->evaluate() * read_b_);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr) and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d PointToPointErrorEval::evaluate(const Eigen::Matrix4d& lhs, std::vector<Jacobian<4,6> >* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation tree
  EvalTreeHandle<lgmath::se3::Transformation> blkAutoTransform =
      T_ab_->getBlockAutomaticEvaluation();

  // Get evaluation from tree
	//TODO: why just not using T_ab_->evaluate() instead?
  const lgmath::se3::Transformation T_ab = blkAutoTransform.getValue();
	const Eigen::Vector4d read_a = T_ab * read_b_;

  // Get Jacobians
	//TODO: why point2fs(...) doesn't take Vector4d as input?
	//TODO: use (-) instead of (-1)
	//TODO: why is the newLhs not the same shape as lhs (input)?
  Eigen::Matrix<double, 4, 6> newLhs = (-1)*lhs*lgmath::se3::point2fs(read_a.head<3>());
  T_ab_->appendBlockAutomaticJacobians(newLhs, blkAutoTransform.getRoot(), jacs);

  // Return evaluation
  return ref_a_ - read_a;
}


} // steam
