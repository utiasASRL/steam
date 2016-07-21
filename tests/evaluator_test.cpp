#include "catch.hpp"
#include "steam/steam.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////
/// ErrorEvaluator Test
/// \author Francois Pomerleau
/////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Vector4d initVector4d(double a,double b, double c, double d)
{
	Eigen::Vector4d v;
	v << a, b, c, d;

	return v;
}




TEST_CASE("PointToPointErrorEval", "[ErrorEvaluator]" ) {

	//---------------------------------------
	// 1- Steam state variables
	//---------------------------------------

	// Setup state for the reference frame
	//TODO: add identity here
	steam::se3::TransformStateVar::Ptr stateReference(new steam::se3::TransformStateVar());
	// Lock the reference frame
	stateReference->setLock(true);

	// Setup state for the reading frame
	steam::se3::TransformStateVar::Ptr stateReading(new steam::se3::TransformStateVar());

	//---------------------------------------
	// 2- Steam cost terms
	//---------------------------------------

	// steam cost terms
	steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());


	// Set the NoiseModel (R_i) to identity
	steam::BaseNoiseModel<4>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<4>(Eigen::MatrixXd::Identity(4,4)));

	// Set the LossFunction to L2 (least-squared)
	steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

	// Build simple point cloud
	const Eigen::Vector4d ref_0 = initVector4d(0, 0, 0, 1);
	const Eigen::Vector4d ref_1 = initVector4d(1, 0, 0, 1);
	const Eigen::Vector4d ref_2 = initVector4d(0, 1, 0, 1);

	Eigen::Matrix4d T_gt;
	T_gt << 1, 0, 0, 1,
	        0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1;

	Eigen::Vector4d read_0 = T_gt * ref_0;
	Eigen::Vector4d read_1 = T_gt * ref_1;
	Eigen::Vector4d read_2 = T_gt * ref_2;

	auto sharedStateReference = steam::se3::TransformStateEvaluator::MakeShared(stateReference);
	auto sharedStateReading = steam::se3::TransformStateEvaluator::MakeShared(stateReading);
	

	steam::PointToPointErrorEval::Ptr errorfunc_0(new steam::PointToPointErrorEval(ref_0, sharedStateReference, read_0, sharedStateReading));
	steam::PointToPointErrorEval::Ptr errorfunc_1(new steam::PointToPointErrorEval(ref_1, sharedStateReference, read_1, sharedStateReading));
	steam::PointToPointErrorEval::Ptr errorfunc_2(new steam::PointToPointErrorEval(ref_2, sharedStateReference, read_2, sharedStateReading));

	steam::WeightedLeastSqCostTerm<4,6>::Ptr cost_0(new steam::WeightedLeastSqCostTerm<4,6>(errorfunc_0, sharedNoiseModel, sharedLossFunc));
	steam::WeightedLeastSqCostTerm<4,6>::Ptr cost_1(new steam::WeightedLeastSqCostTerm<4,6>(errorfunc_1, sharedNoiseModel, sharedLossFunc));
	steam::WeightedLeastSqCostTerm<4,6>::Ptr cost_2(new steam::WeightedLeastSqCostTerm<4,6>(errorfunc_2, sharedNoiseModel, sharedLossFunc));
	
	
	costTerms->add(cost_0);
	costTerms->add(cost_1);
	costTerms->add(cost_2);


	//---------------------------------------
	// 3- Optimization problem
	//---------------------------------------

	steam::OptimizationProblem problem;

	// Add states
	problem.addStateVariable(stateReference);
	problem.addStateVariable(stateReading);

	// Add cost terms
	problem.addCostTerm(costTerms);

	//---------------------------------------
	// 4- Solve
	//---------------------------------------

	//typedef steam::DoglegGaussNewtonSolver SolverType
	typedef steam::VanillaGaussNewtonSolver SolverType;

	SolverType::Params params;
	params.verbose = true;
	//params.maxIterations = 500;
	//params.absoluteCostThreshold = 0.0;
	//params.absoluteCostChangeThreshold = 1e-4;
	//params.relativeCostChangeThreshold = 1e-4;

	// Make solver
	SolverType solver(&problem, params);

	// Optimize
	solver.optimize();

	std::cout << stateReading->getValue() << std::endl;

} // TEST_CASE
