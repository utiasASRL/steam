//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SimpleP2PandCATrajPrior.cpp
/// \brief A sample usage of the STEAM Engine library for a bundle adjustment problem
///        with point-to-point error terms and trajectory smoothing factors.
///
/// \author Tim Tang, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <steam/data/ParseBA.hpp>
#include <fstream>

using namespace std;
typedef steam::PointToPointErrorEval Error; 
typedef steam::WeightedLeastSqCostTerm<4,6> Cost;
unsigned int M = 20;
unsigned int N = 20;

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Structure to store trajectory state variables
//////////////////////////////////////////////////////////////////////////////////////////////
struct TrajStateVar {
  steam::Time time;
  steam::se3::TransformStateVar::Ptr pose;
  steam::VectorSpaceStateVar::Ptr velocity;
  steam::VectorSpaceStateVar::Ptr acceleration;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example that loads and solves simple bundle adjustment problems
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  ///
  /// Setup Traj Prior
  ///


  // Smoothing factor diagonal -- in this example, we penalize accelerations in each dimension
  //                              except for the forward and yaw (this should be fairly typical)
  double lin_acc_stddev_x = 1.00; // body-centric (e.g. x is usually forward)
  double lin_acc_stddev_y = 0.01; // body-centric (e.g. y is usually side-slip)
  double lin_acc_stddev_z = 0.01; // body-centric (e.g. z is usually 'jump')
  double ang_acc_stddev_x = 0.01; // ~roll
  double ang_acc_stddev_y = 0.01; // ~pitch
  double ang_acc_stddev_z = 1.00; // ~yaw
  Eigen::Array<double,1,6> Qc_diag;
  Qc_diag << lin_acc_stddev_x, lin_acc_stddev_y, lin_acc_stddev_z,
             ang_acc_stddev_x, ang_acc_stddev_y, ang_acc_stddev_z;
  Qc_diag = Qc_diag * 1;

  // Make Qc_inv
  Eigen::Matrix<double,6,6> Qc_inv; Qc_inv.setZero();
    Qc_inv.diagonal() = 1.0/Qc_diag;

  ///
  /// Parse Dataset
  ///

  string filePath = "../../include/steam/data/PointToPoint/points_simulated.txt";
  ifstream in(filePath);

  vector<vector<double>> fields;
	int count = 0;
  if (in) {
    string line;
    while (getline(in, line)) {
      count++;
      stringstream sep(line);
      string field;
      fields.push_back(vector<double>());
      while (getline(sep, field, ',')) {
          fields.back().push_back(stod(field));
      }
    }
  }

  filePath = "../../include/steam/data/PointToPoint/poses_ic.txt";
  ifstream inIc(filePath);

  vector<vector<double>> fieldsIC;
	count = 0;
  if (inIc) {
    string line;
    while (getline(inIc, line)) {
      count++;
      stringstream sep(line);
      string field;
      fieldsIC.push_back(vector<double>());
      while (getline(sep, field, ',')) {
        fieldsIC.back().push_back(stod(field));
      }
    }
  }

  cout << fieldsIC.size() << endl;
  cout << fieldsIC.back().front() << endl;

  // State variable containers (and related data)
  std::vector<TrajStateVar> traj_states_ic;

  // Zero velocity
  Eigen::Matrix<double,6,1> initVelocity; initVelocity.setZero();
  // initVelocity << -10.0, 0, 0, 0, 0, 0;
  
  // Zero acceleration
  Eigen::Matrix<double,6,1> initAcceleration; initAcceleration.setZero();
  // initAcceleration << -3.0, 0, 0, 0, 0, 0;

  // Setup state variables using initial condition
  for (unsigned int i = 0; i < N; i++) {
    TrajStateVar temp;
    double dt = double(i-0)*0.1;
    temp.time = steam::Time(dt);
    Eigen::Matrix<double,4,4> T_k0;
    T_k0 << fieldsIC[i][0], fieldsIC[i][1], fieldsIC[i][2], fieldsIC[i][3], 
    fieldsIC[i][4], fieldsIC[i][5], fieldsIC[i][6], fieldsIC[i][7], 
    fieldsIC[i][8], fieldsIC[i][9], fieldsIC[i][10], fieldsIC[i][11], 
    0.0, 0.0, 0.0, 1;

    lgmath::se3::Transformation T_k0_tf(T_k0);

    temp.pose = steam::se3::TransformStateVar::Ptr(new steam::se3::TransformStateVar(T_k0_tf));
    initVelocity << fieldsIC[i+20][0],fieldsIC[i+20][1],fieldsIC[i+20][2],fieldsIC[i+20][3],fieldsIC[i+20][4],fieldsIC[i+20][5];
    initAcceleration << fieldsIC[i+20][6],fieldsIC[i+20][7],fieldsIC[i+20][8],fieldsIC[i+20][9],fieldsIC[i+20][10],fieldsIC[i+20][11];
    cout << initVelocity.transpose() << endl;

    temp.velocity = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initVelocity));
    temp.acceleration = steam::VectorSpaceStateVar::Ptr(new steam::VectorSpaceStateVar(initAcceleration));
    traj_states_ic.push_back(temp);
  }

  // Setup Trajectory
  steam::se3::SteamCATrajInterface traj(Qc_inv);
  for (unsigned int i = 0; i < traj_states_ic.size(); i++) {
    TrajStateVar& state = traj_states_ic.at(i);
    steam::se3::TransformStateEvaluator::Ptr temp =
        steam::se3::TransformStateEvaluator::MakeShared(state.pose);
    traj.add(state.time, temp, state.velocity, state.acceleration);
  }

  // Lock first pose (otherwise entire solution is 'floating')
  //  **Note: alternatively we could add a prior to the first pose.
  traj_states_ic[0].pose->setLock(true);
  

  // steam cost terms
  steam::ParallelizedCostTermCollection::Ptr measCostTerms(new steam::ParallelizedCostTermCollection());
  
  // Setup shared noise and loss function
  Eigen::Matrix4d measurementNoise;
  measurementNoise.setIdentity();
  steam::BaseNoiseModel<4>::Ptr sharedNoiseModel(new steam::StaticNoiseModel<4>(measurementNoise));
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());
  Error::Ptr error;
  Cost::Ptr cost;

  // cout << fields[1*N+1][1] << endl;

  for (unsigned int i = 1; i < N; i++) {
    steam::se3::TransformEvaluator::Ptr T_k0 = steam::se3::TransformStateEvaluator::MakeShared(traj_states_ic[i].pose);
    
    for (unsigned int j = 0; j < M; j++) {
      Eigen::Vector4d ref_a;
      ref_a << fields[j][0] , fields[j][1] , fields[j][2] , fields[j][3];

      Eigen::Vector4d read_b;
      read_b << fields[i*N+j][0] , fields[i*N+j][1] , fields[i*N+j][2] , fields[i*N+j][3];
      
      error.reset(new Error(ref_a, read_b, T_k0));
			cost.reset(new Cost(error, sharedNoiseModel, sharedLossFunc));
      measCostTerms->add(cost);
    }
  }

  // Trajectory prior smoothing terms
  steam::ParallelizedCostTermCollection::Ptr smoothingCostTerms(new steam::ParallelizedCostTermCollection());
  traj.appendPriorCostTerms(smoothingCostTerms);

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < traj_states_ic.size(); i++) {
    const TrajStateVar& state = traj_states_ic.at(i);
    problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
    problem.addStateVariable(state.acceleration);
  }

  // Add cost terms
  problem.addCostTerm(measCostTerms);
  problem.addCostTerm(smoothingCostTerms);

  ///
  /// Setup Solver and Optimize
  ///
  typedef steam::VanillaGaussNewtonSolver SolverType;

  // Initialize parameters (enable verbose mode)
  SolverType::Params params;
  params.verbose = true;
  params.absoluteCostChangeThreshold = 1e-20;
	params.relativeCostChangeThreshold = 1e-20;

  // Make solver
  SolverType solver(&problem, params);

  // Optimize
  TrajStateVar& state = traj_states_ic.back();
  cout << state.velocity->getValue().transpose() << endl;
  cout << state.acceleration->getValue().transpose() << endl;

  solver.optimize();

  // Setup Trajectory
  // for (unsigned int i = 0; i < traj_states_ic.size(); i++) {
  //   TrajStateVar& state = traj_states_ic.at(i);
  //   std::cout << i << ": \n " << state.pose->getValue() << "\n";
  // }

  state = traj_states_ic.back();
  cout << state.pose->getValue() << endl;
  cout << state.velocity->getValue().transpose() << endl;
  cout << state.acceleration->getValue().transpose() << endl;
  return 0;

}
