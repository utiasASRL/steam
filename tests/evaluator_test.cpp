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

// Helper function to solve and check that the solution is close
// to the ground truth transformation used to generate the data
void solveSimpleProblemTrajectory(const Eigen::Matrix<double, 6, 1> T_a_0_components, 
								  const Eigen::Matrix<double, 6, 1> T_b_0_components, 
								  const Eigen::Matrix<double, 6, 1> T_1_0_components,
								  const Eigen::Matrix<double, 6, 1> T_2_0_components,
								  const Eigen::Matrix<double, 6, 1> T_8_0_components, 
								  const Eigen::Matrix<double, 6, 1> T_9_0_components, 
								  const Eigen::Matrix<double, 6, 1> T_10_0_components, 
								  const int constructor_type)
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

	// Number of points in each point cloud
	unsigned int num_pts = 8;

	// Build a fixed point cloud (reference) with 8 points
	// expressed in frame a
	std::vector<Eigen::Vector4d> ref_a_pts;
	for (unsigned int i = 0; i < num_pts; i++) {
		const Eigen::Vector3d rand = Eigen::Matrix<double, 3, 1>::Random();
		const Eigen::Vector4d ref_a_temp = initVector4d(rand(0,0), rand(1,0), rand(2,0));
		ref_a_pts.push_back(ref_a_temp);
	}

	// Build a ground truth (gt) transformation matrix 
	// from frame b to frame a
	const lgmath::se3::Transformation Tgt_a_0(T_a_0_components);
	const lgmath::se3::Transformation Tgt_b_0(T_b_0_components);
	const lgmath::se3::Transformation Tgt_a_b = Tgt_a_0 * Tgt_b_0.inverse();
	// and its inverse
	const lgmath::se3::Transformation Tgt_b_a = Tgt_a_b.inverse();

	// Build ground truth for transformation between key frames
	const lgmath::se3::Transformation Tgt_0_0;
	const lgmath::se3::Transformation Tgt_1_0(T_1_0_components); 
	const lgmath::se3::Transformation Tgt_2_0(T_2_0_components); 
	const lgmath::se3::Transformation Tgt_8_0(T_8_0_components); 
	const lgmath::se3::Transformation Tgt_9_0(T_9_0_components); 
	const lgmath::se3::Transformation Tgt_10_0(T_10_0_components); 
  
	// Move the reference point cloud to generate a
	// second point cloud called read (reading) (still ground truth)
	std::vector<Eigen::Vector4d> read_b_pts;
	for (unsigned int i = 0; i < num_pts; i++) {
		const Eigen::Vector4d ref_a_temp = ref_a_pts.at(i);
		
		// T_temp is the transformation of the sensor between when point in reading frame
		// was taken and when the corresponding point in reference frame was taken

		// For each pair of points, the point in reference is taken at time i, the point
		// in reading is taken at time j.
		Eigen::Matrix4d T_a_i;
		Eigen::Matrix4d T_j_b;
		Eigen::Matrix4d T_j_i;
		if (i < 2){
			T_a_i = Tgt_a_0.matrix() * Tgt_0_0.matrix();
			T_j_b = Tgt_8_0.matrix() * Tgt_b_0.inverse().matrix();
		}
		else if (i >= 2 and i < 5){
			T_a_i = Tgt_a_0.matrix() * Tgt_1_0.inverse().matrix();
			T_j_b = Tgt_9_0.matrix() * Tgt_b_0.inverse().matrix();
		}
		else {
			T_a_i = Tgt_a_0.matrix() * Tgt_2_0.inverse().matrix();
			T_j_b = Tgt_10_0.matrix() * Tgt_b_0.inverse().matrix();
		}
		T_j_i = T_j_b * Tgt_b_a.matrix() * T_a_i;
		Eigen::Vector4d read_b_temp = T_j_i * ref_a_temp;
		read_b_pts.push_back(read_b_temp);
	}

	
	//---------------------------------------
	// 1- Steam state variables
	//---------------------------------------

	// State variable containers (and related data)
  	std::vector<steam::se3::SteamTrajVar> traj_states_ic;
	std::vector<steam::se3::TransformStateVar::Ptr> statevars_ic;

	// Setup trajectory state variables using initial condition. We have 6 trajectory state variables
	// Corresponding to T_0_0, T_1_0, T_2_0, T_8_0, T_9_0, T_10_0

	double time_array[6] = {10000.0, 10001.0, 10002.0, 10008.0, 10009.0, 10010.0};

	for (int i = 0; i < 6; i++) {
		double time = time_array[i];
		steam::Time temp_time = time;
		steam::se3::TransformStateVar::Ptr temp_statevar(new steam::se3::TransformStateVar());
		steam::se3::TransformStateEvaluator::Ptr temp_pose = steam::se3::TransformStateEvaluator::MakeShared(temp_statevar);
		steam::VectorSpaceStateVar::Ptr temp_velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
		steam::se3::SteamTrajVar temp(temp_time, temp_pose, temp_velocity);

		statevars_ic.push_back(temp_statevar);
		traj_states_ic.push_back(temp);
	}

	steam::se3::TransformStateVar::Ptr T_a_0_statevar(new steam::se3::TransformStateVar());
	steam::se3::TransformStateVar::Ptr T_b_0_statevar(new steam::se3::TransformStateVar());

	statevars_ic.push_back(T_a_0_statevar);
	statevars_ic.push_back(T_b_0_statevar);

	// Lock the first state variable
	statevars_ic[0]->setLock(true);

	// Convert our states to Transform Evaluators
	steam::se3::TransformStateEvaluator::Ptr T_a_0 = steam::se3::TransformStateEvaluator::MakeShared(T_a_0_statevar);
	steam::se3::TransformStateEvaluator::Ptr T_b_0 = steam::se3::TransformStateEvaluator::MakeShared(T_b_0_statevar);
	steam::se3::TransformEvaluator::Ptr T_a_b = steam::se3::composeInverse(T_a_0, T_b_0);
	steam::se3::TransformStateEvaluator::Ptr T_0_0 = steam::se3::TransformStateEvaluator::MakeShared(statevars_ic[0]);

	// Setup Trajectory
	steam::se3::SteamTrajInterface traj(Qc_inv);
	for (unsigned int i = 0; i < traj_states_ic.size(); i++) {
		steam::se3::SteamTrajVar& state = traj_states_ic.at(i);
		steam::Time temp_time = state.getTime();
		steam::se3::TransformEvaluator::Ptr temp_pose = state.getPose();
		steam::VectorSpaceStateVar::Ptr temp_velocity = state.getVelocity();
		traj.add(temp_time, temp_pose, temp_velocity);
	}

	// Define our error funtion
	typedef steam::PointToPointErrorEval Error;
	
	//-------------
	// This is spestateReferencecific to the unit test

	// Build the alignment errors
	std::vector<Error::Ptr> errors;

	for (unsigned int i = 0; i < ref_a_pts.size(); i++) {
		Error::Ptr error_temp;
		const Eigen::Vector4d ref_a_temp = ref_a_pts.at(i);
		Eigen::Vector4d read_b_temp = read_b_pts.at(i);

		steam::se3::TransformEvaluator::Ptr T_i_0;
		steam::se3::TransformEvaluator::Ptr T_j_0;

		if (i < 2){
			T_i_0 = traj_states_ic[0].getPose();
			T_j_0 = traj_states_ic[3].getPose();
		}
		else if (i >= 2 and i < 5){
			T_i_0 = traj_states_ic[1].getPose();
			T_j_0 = traj_states_ic[4].getPose();
		}
		else {
			T_i_0 = traj_states_ic[2].getPose();
			T_j_0 = traj_states_ic[5].getPose();
		}

		steam::se3::TransformEvaluator::Ptr T_i_a = steam::se3::composeInverse(T_i_0, T_a_0);
		steam::se3::TransformEvaluator::Ptr T_b_j = steam::se3::composeInverse(T_b_0, T_j_0);
		steam::se3::TransformEvaluator::Ptr T_i_b = steam::se3::compose(T_i_a, T_a_b);
		steam::se3::TransformEvaluator::Ptr T_i_j = steam::se3::compose(T_i_b, T_b_j);
		
		if(constructor_type == 0)
		{
			error_temp.reset(new Error(ref_a_temp, T_0_0, read_b_temp, T_i_j)); 
		}
		else if(constructor_type == 1)
		{
			error_temp.reset(new Error(ref_a_temp, read_b_temp, T_a_b));
		}
		errors.push_back(error_temp);
	}

	
	//---------------------------------------
	// 2- Steam cost terms
	//---------------------------------------

	// steam cost terms
	steam::ParallelizedCostTermCollection::Ptr costTerms(new steam::ParallelizedCostTermCollection());

	// Set the NoiseModel (R_i) to identity
	steam::BaseNoiseModel<4>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<4>(Eigen::Matrix4d::Identity()));

	// Set the LossFunction to L2 (least-squared)
	steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

	// Define our cost term
	typedef steam::WeightedLeastSqCostTerm<4,6> Cost;

	// Build the cost terms. Add our individual cost terms to the collection 
	std::vector<Cost::Ptr> costs;
	for (unsigned int i = 0; i < errors.size(); i++) {
		Error::Ptr error_temp = errors.at(i);
		Cost::Ptr cost_temp(new Cost(error_temp, sharedNoiseModel, sharedLossFunc));
		costs.push_back(cost_temp);
		costTerms->add(cost_temp);
	}

	// Trajectory prior smoothing terms
	steam::ParallelizedCostTermCollection::Ptr smoothingCostTerms(new steam::ParallelizedCostTermCollection());
	traj.appendPriorCostTerms(smoothingCostTerms);

	//---------------------------------------
	// 3- Set up the optimization problem
	//---------------------------------------

	steam::OptimizationProblem problem;

	// Add state variables

	for (unsigned int i = 0; i < traj_states_ic.size(); i++) {
		steam::se3::SteamTrajVar& state = traj_states_ic.at(i);
		steam::VectorSpaceStateVar::Ptr temp_velocity = state.getVelocity();
		problem.addStateVariable(temp_velocity);
  	}

	for (unsigned int i = 0; i < statevars_ic.size(); i++) {
		steam::se3::TransformStateVar::Ptr& state_var= statevars_ic.at(i);
		problem.addStateVariable(state_var);
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
		 T_a_0_statevar->getValue().matrix() << "\n" <<
		 "is different than original transformation:" << "\n" <<
		 Tgt_a_0.matrix() << "\n" << 
		 "difference being:" << "\n" <<
		 T_a_0_statevar->getValue().matrix() - Tgt_a_0.matrix()
		 );

	// Confirm that our state is the same as our ground truth transformation
	CHECK(lgmath::common::nearEqual(T_a_0_statevar->getValue().matrix(), 
								  	Tgt_a_0.matrix(),
								  	1e-3
								  	));

}

void solveSimpleProblem(const Eigen::Matrix<double, 6, 1> T_components, const int constructor_type)
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
	auto T_b_a = steam::se3::InverseTransformEvaluator::MakeShared(T_a_b);

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
	Eigen::Matrix<double, 6, 1> T_a_0_components;
	Eigen::Matrix<double, 6, 1> T_b_0_components;
	Eigen::Matrix<double, 6, 1> T_1_0_components;
	Eigen::Matrix<double, 6, 1> T_2_0_components;
	Eigen::Matrix<double, 6, 1> T_8_0_components;
	Eigen::Matrix<double, 6, 1> T_9_0_components;
	Eigen::Matrix<double, 6, 1> T_10_0_components;

	SECTION("Simple translation - constructor 0")
	{
		T_components << 1.0, // translation x
						0.0, // translation y
						0.0, // translation z
						0.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis
		solveSimpleProblemTrajectory(T_components, T_components, T_components, T_components, T_components, T_components, T_components, 0);
		// solveSimpleProblem(T_components, 0);
	}
	
	SECTION("Simple rotation - constructor 0")
	{
		T_components << 0.0, // translation x
						0.0, // translation y
						0.0, // translation z
						1.0, // rotation around x-axis
						0.0, // rotation around y-axis
						0.0; // rotation around z-axis
		solveSimpleProblemTrajectory(T_components, T_components, T_components, T_components, T_components, T_components, T_components, 0);				
		// solveSimpleProblem(T_components, 0);
	}
	
	SECTION("Random transformation (1000) - constructor 0")
	{
		srand((unsigned int) time(0));

		for(int i=0; i<1000; i++)
		{
			// random numbers in interval [-1, 1]
			T_a_0_components = Eigen::Matrix<double, 6, 1>::Random();
			T_b_0_components = Eigen::Matrix<double, 6, 1>::Random();
			T_1_0_components = Eigen::Matrix<double, 6, 1>::Random();
			T_2_0_components = Eigen::Matrix<double, 6, 1>::Random();
			T_8_0_components = Eigen::Matrix<double, 6, 1>::Random();
			T_9_0_components = Eigen::Matrix<double, 6, 1>::Random();
			T_10_0_components = Eigen::Matrix<double, 6, 1>::Random();
			solveSimpleProblemTrajectory(T_a_0_components, T_b_0_components, T_1_0_components, T_2_0_components, T_8_0_components, T_9_0_components, T_10_0_components, 0);
			// solveSimpleProblem(T_components, 0);
		}

	}
	
	// SECTION("Simple translation - constructor 1")
	// {
	// 	T_components << 1.0, // translation x
	// 					0.0, // translation y
	// 					0.0, // translation z
	// 					0.0, // rotation around x-axis
	// 					0.0, // rotation around y-axis
	// 					0.0; // rotation around z-axis
	// 	solveSimpleProblemTrajectory(T_components, 0);				
	// 	// solveSimpleProblem(T_components, 1);
	// }
	
	// SECTION("Simple rotation - constructor 1")
	// {
	// 	T_components << 0.0, // translation x
	// 					0.0, // translation y
	// 					0.0, // translation z
	// 					1.0, // rotation around x-axis
	// 					0.0, // rotation around y-axis
	// 					0.0; // rotation around z-axis
	// 	solveSimpleProblemTrajectory(T_components, 0);			
	// 	// solveSimpleProblem(T_components, 1);
	// }
	
	// SECTION("Random transformation (1000) - constructor 1")
	// {
	// 	srand((unsigned int) time(0));

	// 	for(int i=0; i<1000; i++)
	// 	{
	// 		// random numbers in interval [-1, 1]
	// 		T_components = Eigen::Matrix<double, 6, 1>::Random();
	// 		solveSimpleProblemTrajectory(T_components, 0);
	// 		// solveSimpleProblem(T_components, 1);
	// 	}

	// }

} // TEST_CASE
