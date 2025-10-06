/**
 * \file SimpleConstVel2DTrajPrior.cpp
 * \author Daniil Lisus, Autonomous Space Robotics Lab (ASRL)
 * \brief A sample usage of the STEAM Engine library for a 2D trajectory prior
 * problem.
 */
#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace steam;
using namespace steam::traj;

/** \brief Structure to store trajectory state variables */
struct TrajStateVar {
  Time time;
  se2::SE2StateVar::Ptr pose;
  vspace::VSpaceStateVar<3>::Ptr velocity;
};

/** \brief Example that uses a constant-velocity prior over a trajectory. */
int main(int argc, char** argv) {
  ///
  /// Setup Problem
  ///

  // Number of state times
  unsigned int numPoses = 100;

  // Setup velocity prior
  Eigen::Matrix<double, 3, 1> velocityPrior;
  double v_x = 5.0;
  double omega_z = 0.01;
  velocityPrior << v_x, 0.0, omega_z;

  // Calculate time to do one circle
  double totalTime = 2.0 * M_PI / omega_z;

  // Calculate time between states
  double delT = totalTime / (numPoses - 1);

  // Smoothing factor diagonal
  Eigen::Array<double, 1, 3> Qc_diag;
  Qc_diag << 1.0, 0.001, 1.0;

  //
  // Setup initial conditions
  //

  // Pose
  Eigen::Matrix<double, 3, 1> initPoseVec;
  initPoseVec << 1, 2, 3;
  lgmath::se2::Transformation initPose(initPoseVec);

  // Zero velocity
  Eigen::Matrix<double, 3, 1> initVelocity;
  initVelocity.setZero();

  ///
  /// Setup States
  ///

  // Steam state variables
  std::vector<TrajStateVar> states;

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 0; i < numPoses; i++) {
    TrajStateVar temp;
    temp.time = Time(i * delT);
    temp.pose = se2::SE2StateVar::MakeShared(initPose);
    temp.velocity = vspace::VSpaceStateVar<3>::MakeShared(initVelocity);
    states.push_back(temp);
  }

  // Setup Trajectory
  const_vel_2d::Interface traj(Qc_diag);
  for (const auto& state : states)
    traj.add(state.time, state.pose, state.velocity);

  // Add priors (alternatively, we could lock pose variable)
  traj.addPosePrior(Time(0.0), initPose,
                    Eigen::Matrix<double, 3, 3>::Identity());
  traj.addVelocityPrior(Time(0.0), velocityPrior,
                        Eigen::Matrix<double, 3, 3>::Identity());

  ///
  /// Make Optimization Problem
  ///

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < states.size(); i++) {
    const TrajStateVar& state = states.at(i);
    problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }

  // Add cost terms
  traj.addPriorCostTerms(problem);

  ///
  /// Setup Solver and Optimize
  ///
  steam::DoglegGaussNewtonSolver::Params params;
  params.verbose = true;
  steam::DoglegGaussNewtonSolver solver(problem, params);

  // Optimize
  solver.optimize();

  // Get velocity at interpolated time
  auto curr_vel =
      traj.getVelocityInterpolator(states.at(0).time + 0.5 * delT)->evaluate();

  ///
  /// Print results
  ///
  // clang-format off
  std::cout << std::endl
            << "First Pose:                  " << states.at(0).pose->value()
            << "Second Pose:                 " << states.at(1).pose->value()
            << "Last Pose (full circle):     " << states.back().pose->value()
            << "First Vel:                   " << states.at(0).velocity->value().transpose() << std::endl
            << "Second Vel:                  " << states.at(1).velocity->value().transpose() << std::endl
            << "Last Vel:                    " << states.back().velocity->value().transpose() << std::endl
            << "Interp. Vel (t=t0+0.5*delT): " << curr_vel.transpose() << std::endl;
  // clang-format on

  return 0;
}
