// vim: ts=4:sw=4:noexpandtab

/////////////////////////////////////////////////////////////////////////////////////////////
/// ErrorEvaluator Test
/// \author Francois Pomerleau
/// \brief To execute only this test run: 
///    $ ./tests/steam_unit_tests PointToPointErrorEval -s
/////////////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"
#include "steam/steam.hpp"
#include <lgmath/CommonMath.hpp>

// Helper function to initialize a 3D point in homogeneous coordinates 
Eigen::Vector4d initVector4d(double x, double y, double z)
{
	Eigen::Vector4d v;
	v << x, y, z, 1.0;

	return v;
}

// Helper function to solve and check that the solution is close
// to the ground truth transformation used to generate the data
void solveSimpleProblem(Eigen::Matrix<double, 6, 1> T_components)
{
	//---------------------------------------
	// General structure:
	// 0- Generate simple point clouds
	// 1- Steam state variables
	// 2- Steam cost terms
	// 3- Set up the optimization problem
	// 4- Solve
	// 5- Check that the transformation are the same
	//---------------------------------------


	//---------------------------------------
	// 0- Generate simple point clouds
	//---------------------------------------

	// Build a fixed point cloud (reference) with 3 points
	// expressed in frame a
	const Eigen::Vector4d ref_a_0 = initVector4d(0, 0, 0);
	const Eigen::Vector4d ref_a_1 = initVector4d(1, 0, 0);
	const Eigen::Vector4d ref_a_2 = initVector4d(0, 1, 0);

	// Build a ground truth (gt) transformation matrix 
	// from frame b to frame a
	const lgmath::se3::Transformation Tgt_a_b(T_components);
	// and its inverse
	const lgmath::se3::Transformation Tgt_b_a = Tgt_a_b.inverse();
  

	// Move the reference point cloud to generate a
	// second point cloud called read (reading)
	Eigen::Vector4d read_b_0 = Tgt_b_a.matrix() * ref_a_0;
	Eigen::Vector4d read_b_1 = Tgt_b_a.matrix() * ref_a_1;
	Eigen::Vector4d read_b_2 = Tgt_b_a.matrix() * ref_a_2;
	
	//---------------------------------------
	// 1- Steam state variables
	//---------------------------------------

	// Setup state for the reference frame
	// Note: the default constructor sets the state to identity
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

	// Convert our states to Transform Evaluators
	// T_a_a is silly (most be identity) but it's there for completness
	auto T_a_a = steam::se3::TransformStateEvaluator::MakeShared(stateReference);
	auto T_a_b = steam::se3::TransformStateEvaluator::MakeShared(stateReading);

	// Define our error funtion
	typedef steam::PointToPointErrorEval Error;
	
	// Build the alignment errors
	Error::Ptr error_0(new Error(ref_a_0, T_a_a, read_b_0, T_a_b));
	Error::Ptr error_1(new Error(ref_a_1, T_a_a, read_b_1, T_a_b));
	Error::Ptr error_2(new Error(ref_a_2, T_a_a, read_b_2, T_a_b));

		
	// Set the NoiseModel (R_i) to identity
	steam::BaseNoiseModel<4>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<4>(Eigen::Matrix4d::Identity()));

	// Set the LossFunction to L2 (least-squared)
	steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

	// Define our cost term
	typedef steam::WeightedLeastSqCostTerm<4,6> Cost;

	// Build the cost terms
	Cost::Ptr cost_0(new Cost(error_0, sharedNoiseModel, sharedLossFunc));
	Cost::Ptr cost_1(new Cost(error_1, sharedNoiseModel, sharedLossFunc));
	Cost::Ptr cost_2(new Cost(error_2, sharedNoiseModel, sharedLossFunc));
	
	// Add our individual cost terms to the collection 
	costTerms->add(cost_0);
	costTerms->add(cost_1);
	costTerms->add(cost_2);


	//---------------------------------------
	// 3- Set up the optimization problem
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
	params.verbose = false;
	//params.maxIterations = 500;
	//params.absoluteCostThreshold = 0.0;
	//params.absoluteCostChangeThreshold = 1e-4;
	//params.relativeCostChangeThreshold = 1e-4;

	// Make solver
	SolverType solver(&problem, params);

	// Optimize
	solver.optimize();

	//---------------------------------------
	// 5- Check that the transformation are the same
	//---------------------------------------
	INFO("Minimized transformation:" << "\n" <<
		 stateReading->getValue().matrix() << "\n" <<
		 "is different than original transformation:" << "\n" <<
		 Tgt_b_a.matrix() << "\n" << 
		 "difference being:" << "\n" <<
		 stateReading->getValue().matrix() - Tgt_b_a.matrix()
		 );

	// Confirm that our state is the same as our ground truth transformation
	CHECK(lgmath::common::nearEqual(stateReading->getValue().matrix(), 
								  	Tgt_b_a.matrix(),
								  	1e-3
								  	));

}


TEST_CASE("PointToPointErrorEval", "[ErrorEvaluator]" ) {
	
	Eigen::Matrix<double, 6, 1> T_components;

	SECTION("Simple translation")
	{
		T_components << 1.0, // translation x
						0.0, // translation y
						0.0, // translation z
						0.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis

		solveSimpleProblem(T_components);
	}
	
	SECTION("Simple rotation")
	{
		T_components << 0.0, // translation x
						0.0, // translation y
						0.0, // translation z
						1.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis

		solveSimpleProblem(T_components);
	}
	
	SECTION("Random transformation (1000)")
	{
		srand((unsigned int) time(0));

		for(int i=0; i<1000; i++)
		{
			// random numbers in interval [-1, 1]
			T_components = Eigen::Matrix<double, 6, 1>::Random();
			solveSimpleProblem(T_components);
		}

	}


} // TEST_CASE
