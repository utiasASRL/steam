// vim: ts=4:sw=4:noexpandtab
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
PointToPointErrorEval::PointToPointErrorEval(
		const Eigen::Vector4d& ref_a,
		const se3::TransformEvaluator::ConstPtr& T_a_world,
		const Eigen::Vector4d& read_b,
		const se3::TransformEvaluator::ConstPtr& T_b_world
		): ref_a_(ref_a),
		   T_b_a_(se3::ComposeInverseTransformEvaluator::MakeShared(T_b_world, T_a_world)),
		   read_b_(read_b)	{
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
PointToPointErrorEval::PointToPointErrorEval(
		const Eigen::Vector4d& ref_a,
		const Eigen::Vector4d& read_b,
		const se3::TransformEvaluator::ConstPtr& T_b_a
		): ref_a_(ref_a),
		   T_b_a_(T_b_a),
		   read_b_(read_b){
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool PointToPointErrorEval::isActive() const {
	return T_b_a_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 4-d measurement error (ul vl ur vr)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d PointToPointErrorEval::evaluate() const {

	// Return error (between measurement and point estimate projected in camera frame)
	// return ref_a_ - (T_b_a_->evaluate().inverse() * read_b_);
	return read_b_ - T_b_a_->evaluate() * ref_a_;
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
		T_b_a_->getBlockAutomaticEvaluation();

	// Get evaluation from tree
	//TODO: why just not using T_a_b_->evaluate() instead?
	const lgmath::se3::Transformation T_ba = blkAutoTransform.getValue();
	// const lgmath::se3::Transformation T_ba = T_b_a_->evaluate();
	// const Eigen::Vector4d read_a = T_ba.inverse() * read_b_;
	const Eigen::Vector4d ref_b = T_ba * ref_a_;

	// Get Jacobians

	// Eigen::Matrix<double, 4, 6> newLhs = -lhs*lgmath::se3::point2fs(read_a.head<3>());
	// Eigen::Matrix<double, 4, 6> newLhs = lhs*T_ba.adjoint()*lgmath::se3::point2fs(read_a.head<3>());
	Eigen::Matrix<double, 4, 6> newLhs = -lhs*lgmath::se3::point2fs(ref_b.head<3>());

	T_b_a_->appendBlockAutomaticJacobians(newLhs, blkAutoTransform.getRoot(), jacs);

	// Return evaluation
	// return ref_a_ - read_a;
	return read_b_ - ref_b;
}


} // steam
