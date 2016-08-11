// vim: ts=4:sw=4:noexpandtab

/////////////////////////////////////////////////////////////////////////////////////////////
/// ErrorEvaluator Test
/// \author Francois Pomerleau
/// \brief To execute only this test run: 
///    $ ./tests/steam_unit_tests PointToPointErrorEval -s
/////////////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"
#include "steam.hpp"
#include <lgmath/CommonMath.hpp>
#include <steam/data/ParseBA.hpp>

// Helper function to initialize a 3D point in homogeneous coordinates 
Eigen::Vector4d initVector4d(const double x, const double y, const double z)
{
	Eigen::Vector4d v;
	v << x, y, z, 1.0;

	return v;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Structure to store trajectory state variables
//////////////////////////////////////////////////////////////////////////////////////////////
struct TrajStateVar {
  steam::Time time;
  steam::se3::TransformStateVar::Ptr pose;
  steam::VectorSpaceStateVar::Ptr velocity;
};

// Helper function to solve and check that the solution is close
// to the ground truth transformation used to generate the data
void solveSimpleProblem(const Eigen::Matrix<double, 6, 1> T_components, const int constructor_type)
{
	//---------------------------------------
	// Preliminary things used for setting up a trajectory

	// Make Qc_inv for the trajectory interface
	double lin_acc_stddev_x = 1.00; // body-centric (e.g. x is usually forward)
	double lin_acc_stddev_y = 0.01; // body-centric (e.g. y is usually side-slip)
	double lin_acc_stddev_z = 0.01; // body-centric (e.g. z is usually 'jump')
	double ang_acc_stddev_x = 0.01; // ~roll
	double ang_acc_stddev_y = 0.01; // ~pitch
	double ang_acc_stddev_z = 1.00; // ~yaw
	Eigen::Array<double,1,6> Qc_diag;
	Qc_diag << lin_acc_stddev_x, lin_acc_stddev_y, lin_acc_stddev_z,
				ang_acc_stddev_x, ang_acc_stddev_y, ang_acc_stddev_z;

	// Make Qc_inv
	Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
		Qc_inv.diagonal() = 1.0/Qc_diag;
	
	// Zero velocity
	Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();


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

	// State variable containers (and related data)
  	std::vector<TrajStateVar> traj_states_ic;

	// Setup trajectory state variables using initial condition
	TrajStateVar TrajStateVar_0;
	TrajStateVar_0.time = 10000.0;
	TrajStateVar_0.pose = stateReference;
	TrajStateVar_0.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
	traj_states_ic.push_back(TrajStateVar_0);

	TrajStateVar TrajStateVar_1;
	TrajStateVar_1.time = 10001.0;
	TrajStateVar_1.pose = stateReading;
	TrajStateVar_1.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
	traj_states_ic.push_back(TrajStateVar_1);


	// Convert our states to Transform Evaluators
	// T_a_a is silly (most be identity) but it's there for completness
	auto T_a_a = steam::se3::TransformStateEvaluator::MakeShared(TrajStateVar_0.pose);
	auto T_a_b = steam::se3::TransformStateEvaluator::MakeShared(TrajStateVar_1.pose);
	auto T_b_a = steam::se3::InverseTransformEvaluator::MakeShared(T_a_b);

	// Setup Trajectory
	steam::se3::SteamTrajInterface traj(Qc_inv);
	for (unsigned int i = 0; i < traj_states_ic.size(); i++) {
		TrajStateVar& state = traj_states_ic.at(i);
		steam::se3::TransformStateEvaluator::Ptr temp =
			steam::se3::TransformStateEvaluator::MakeShared(state.pose);
		traj.add(state.time, temp, state.velocity);
	}

	// Lock first pose (otherwise entire solution is 'floating')
	//  **Note: alternatively we could add a prior to the first pose.
	traj_states_ic[0].pose->setLock(true);

	// Define our error funtion
	typedef steam::PointToPointErrorEval Error;
	
	//-------------
	// This is specific to the unit test

	// Build the alignment errors
	Error::Ptr error_0;
	Error::Ptr error_1;
	Error::Ptr error_2;

	if(constructor_type == 0)
	{
		error_0.reset(new Error(ref_a_0, T_a_a, read_b_0, T_b_a));
		error_1.reset(new Error(ref_a_1, T_a_a, read_b_1, T_b_a));
		error_2.reset(new Error(ref_a_2, T_a_a, read_b_2, T_b_a));
	}
	else if(constructor_type == 1)
	{
		error_0.reset(new Error(ref_a_0, read_b_0, T_a_b));
		error_1.reset(new Error(ref_a_1, read_b_1, T_a_b));
		error_2.reset(new Error(ref_a_2, read_b_2, T_a_b));
	}
	//-------------
		
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

	// Trajectory prior smoothing terms
	steam::ParallelizedCostTermCollection::Ptr smoothingCostTerms(new steam::ParallelizedCostTermCollection());
	traj.appendPriorCostTerms(smoothingCostTerms);

	//---------------------------------------
	// 3- Set up the optimization problem
	//---------------------------------------

	steam::OptimizationProblem problem;

	// Add state variables
	// problem.addStateVariable(stateReference);
	// problem.addStateVariable(stateReading);

	for (unsigned int i = 0; i < traj_states_ic.size(); i++) {
		const TrajStateVar& state = traj_states_ic.at(i);
		problem.addStateVariable(state.pose);
		problem.addStateVariable(state.velocity);
  	}

	// Add cost terms
	problem.addCostTerm(costTerms);
	problem.addCostTerm(smoothingCostTerms);

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
		 stateReading->getValue().matrix() - Tgt_a_b.matrix()
		 );

	// Confirm that our state is the same as our ground truth transformation
	CHECK(lgmath::common::nearEqual(stateReading->getValue().matrix(), 
								  	Tgt_a_b.matrix(),
								  	1e-3
								  	));

}


TEST_CASE("PointToPointErrorEval", "[ErrorEvaluator]" ) {
	
	Eigen::Matrix<double, 6, 1> T_components;

	SECTION("Simple translation - constructor 0")
	{
		T_components << 1.0, // translation x
						0.0, // translation y
						0.0, // translation z
						0.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis

		solveSimpleProblem(T_components, 0);
	}
	
	SECTION("Simple rotation - constructor 0")
	{
		T_components << 0.0, // translation x
						0.0, // translation y
						0.0, // translation z
						1.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis

		solveSimpleProblem(T_components, 0);
	}
	
	SECTION("Random transformation (1000) - constructor 0")
	{
		srand((unsigned int) time(0));

		for(int i=0; i<1000; i++)
		{
			// random numbers in interval [-1, 1]
			T_components = Eigen::Matrix<double, 6, 1>::Random();
			solveSimpleProblem(T_components, 0);
		}

	}
	
	SECTION("Simple translation - constructor 1")
	{
		T_components << 1.0, // translation x
						0.0, // translation y
						0.0, // translation z
						0.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis

		solveSimpleProblem(T_components, 1);
	}
	
	SECTION("Simple rotation - constructor 1")
	{
		T_components << 0.0, // translation x
						0.0, // translation y
						0.0, // translation z
						1.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis

		solveSimpleProblem(T_components, 1);
	}
	
	SECTION("Random transformation (1000) - constructor 1")
	{
		srand((unsigned int) time(0));

		for(int i=0; i<1000; i++)
		{
			// random numbers in interval [-1, 1]
			T_components = Eigen::Matrix<double, 6, 1>::Random();
			solveSimpleProblem(T_components, 1);
		}

	}

} // TEST_CASE
