#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace steam;
using namespace steam::traj;

// clang-format off

struct TrajStateVar {
  TrajStateVar(const Time& time_, const se3::SE3StateVar::Ptr& T_k0_,
               const vspace::VSpaceStateVar<6>::Ptr& w_0k_ink_,
               const vspace::VSpaceStateVar<6>::Ptr& dw_0k_ink_ = nullptr)
      : time(time_), T_k0(T_k0_), w_0k_ink(w_0k_ink_), dw_0k_ink(dw_0k_ink_) {}
  Time time;
  se3::SE3StateVar::Ptr T_k0;
  vspace::VSpaceStateVar<6>::Ptr w_0k_ink;
  vspace::VSpaceStateVar<6>::Ptr dw_0k_ink;
};

void WNOAPrior() {
  std::cout << std::endl << "WNOA Trajectory without measurements:" << std::endl;

  int num_knots = 10;
  double dt = 0.1;

  Eigen::Matrix<double, 6, 1> qcd; qcd << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  const auto trajectory = const_vel::Interface::MakeShared(qcd);
  std::vector<TrajStateVar> traj_state_vars;

  //
  for (int i = 0; i <= num_knots; ++i) {
    Time time(static_cast<double>(i) * dt);
    auto T_k0 = se3::SE3StateVar::MakeShared(lgmath::se3::Transformation());
    auto w_0k_ink = vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());
    traj_state_vars.emplace_back(time, T_k0, w_0k_ink);
    trajectory->add(time, T_k0, w_0k_ink);
  }

  // Add priors (alternatively, we could lock pose variable)
  lgmath::se3::Transformation init_pose;
  Eigen::Matrix<double, 6, 1> init_velocity; init_velocity << -1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
  trajectory->addPosePrior(Time(0.0), init_pose, Eigen::Matrix<double, 6, 6>::Identity());
  trajectory->addVelocityPrior(Time(0.0), init_velocity, Eigen::Matrix<double, 6, 6>::Identity());

  //
  using MeasType = Eigen::Matrix<double, 6, 1>;
  std::vector<std::pair<double, MeasType>> measurements;
  auto &meas1 = measurements.emplace_back(0.0, MeasType::Zero());
  meas1.second <<  0.176405,  0.040016,  0.097874,  0.224089,  0.186756, -0.097728;
  auto &meas2 = measurements.emplace_back(0.25, MeasType::Zero());
  meas2.second << -0.154991, -0.015136, -0.010322,  0.04106 ,  0.014404, -0.104573;
  auto &meas3 = measurements.emplace_back(0.5, MeasType::Zero());
  meas3.second << -0.423896,  0.012168,  0.044386,  0.033367,  0.149408, -0.520516;
  auto &meas4 = measurements.emplace_back(0.75, MeasType::Zero());
  meas4.second << -0.718693, -0.08541 , -0.255299,  0.065362,  0.086444, -0.824217;
  auto &meas5 = measurements.emplace_back(1.0, MeasType::Zero());
  meas5.second << -0.773025, -0.145437,  0.004576, -0.018718,  0.153278, -0.853064;

  std::vector<BaseCostTerm::Ptr> meas_terms;
  for (const auto& measurement : measurements) {
    Time time(measurement.first);
    const auto w_0k_ink = trajectory->getVelocityInterpolator(time);
    const auto error_func = vspace::vspace_error<6>(w_0k_ink, measurement.second);
    const auto noise_model = StaticNoiseModel<6>::MakeShared(Eigen::Matrix<double, 6, 6>::Identity());
    const auto loss_func = L2LossFunc::MakeShared();
    meas_terms.emplace_back(WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func));
  }

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < traj_state_vars.size(); i++) {
    const TrajStateVar& state = traj_state_vars.at(i);
    problem.addStateVariable(state.T_k0);
    problem.addStateVariable(state.w_0k_ink);
  }

  // Add cost terms
  trajectory->addPriorCostTerms(problem);
  for (const auto& meas_term : meas_terms)
    problem.addCostTerm(meas_term);

  // Setup Solver and Optimize
  steam::GaussNewtonSolver::Params params;
  params.verbose = true;
  steam::GaussNewtonSolver solver(problem, params);
  solver.optimize();

  // // dump trajectory
  // const double start_time = traj_state_vars.front().time.seconds();
  // const double end_time = traj_state_vars.back().time.seconds();
  // for (double t = start_time; t <= end_time; t += 0.01) {
  //   const auto T_0k_vec = trajectory->getPoseInterpolator(Time(t))->value().inverse().vec();
  //   const auto w_0k_ink = trajectory->getVelocityInterpolator(Time(t))->value();
  //   std::cout << t << " " << T_0k_vec.transpose() << " " << w_0k_ink.transpose() << std::endl;
  // }
}

void WNOJPrior() {
  std::cout << std::endl << "WNOJ Trajectory without measurements:" << std::endl;

  int num_knots = 10;
  double dt = 0.1;

  Eigen::Matrix<double, 6, 1> qcd; qcd << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  const auto trajectory = const_acc::Interface::MakeShared(qcd);
  std::vector<TrajStateVar> traj_state_vars;

  //
  for (int i = 0; i <= num_knots; ++i) {
    Time time(static_cast<double>(i) * dt);
    auto T_k0 = se3::SE3StateVar::MakeShared(lgmath::se3::Transformation());
    auto w_0k_ink = vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());
    auto dw_0k_ink = vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());
    traj_state_vars.emplace_back(time, T_k0, w_0k_ink, dw_0k_ink);
    trajectory->add(time, T_k0, w_0k_ink, dw_0k_ink);
  }

  // Add priors (alternatively, we could lock pose variable)
  lgmath::se3::Transformation init_pose;
  Eigen::Matrix<double, 6, 1> init_velocity = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> init_acceleration; init_acceleration << -1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
  trajectory->addPosePrior(Time(0.0), init_pose, Eigen::Matrix<double, 6, 6>::Identity());
  trajectory->addVelocityPrior(Time(0.0), init_velocity, Eigen::Matrix<double, 6, 6>::Identity());
  trajectory->addAccelerationPrior(Time(0.0), init_acceleration, Eigen::Matrix<double, 6, 6>::Identity());

  //
  using MeasType = Eigen::Matrix<double, 6, 1>;
  std::vector<std::pair<double, MeasType>> measurements;
  auto &meas1 = measurements.emplace_back(0.0, MeasType::Zero());
  meas1.second <<  0.015495,  0.037816, -0.088779, -0.19808 , -0.034791,  0.015635;
  auto &meas2 = measurements.emplace_back(0.25, MeasType::Zero());
  meas2.second <<  0.123029,  0.120238, -0.038733, -0.03023 , -0.104855, -0.142002;
  auto &meas3 = measurements.emplace_back(0.5, MeasType::Zero());
  meas3.second << -0.170627,  0.195078, -0.050965, -0.043807, -0.12528 ,  0.077749;
  auto &meas4 = measurements.emplace_back(0.75, MeasType::Zero());
  meas4.second << -0.16139 , -0.021274, -0.089547,  0.03869 , -0.051081, -0.118063;
  auto &meas5 = measurements.emplace_back(1.0, MeasType::Zero());
  meas5.second << -0.002818,  0.042833,  0.006652,  0.030247, -0.063432, -0.036274;

  std::vector<BaseCostTerm::Ptr> meas_terms;
  for (const auto& measurement : measurements) {
    Time time(measurement.first);
    const auto w_0k_ink = trajectory->getVelocityInterpolator(time);
    const auto error_func = vspace::vspace_error<6>(w_0k_ink, measurement.second);
    const auto noise_model = StaticNoiseModel<6>::MakeShared(Eigen::Matrix<double, 6, 6>::Identity());
    const auto loss_func = L2LossFunc::MakeShared();
    meas_terms.emplace_back(WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func));
  }

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < traj_state_vars.size(); i++) {
    const TrajStateVar& state = traj_state_vars.at(i);
    problem.addStateVariable(state.T_k0);
    problem.addStateVariable(state.w_0k_ink);
    problem.addStateVariable(state.dw_0k_ink);
  }

  // Add cost terms
  trajectory->addPriorCostTerms(problem);
  for (const auto& meas_term : meas_terms)
    problem.addCostTerm(meas_term);

  // Setup Solver and Optimize
  steam::GaussNewtonSolver::Params params;
  params.verbose = true;
  steam::GaussNewtonSolver solver(problem, params);
  solver.optimize();

  // // dump trajectory
  // const double start_time = traj_state_vars.front().time.seconds();
  // const double end_time = traj_state_vars.back().time.seconds();
  // for (double t = start_time; t <= end_time; t += 0.01) {
  //   const auto T_0k_vec = trajectory->getPoseInterpolator(Time(t))->value().inverse().vec();
  //   const auto w_0k_ink = trajectory->getVelocityInterpolator(Time(t))->value();
  //   std::cout << t << " " << T_0k_vec.transpose() << " " << w_0k_ink.transpose() << std::endl;
  // }
}

void SingerPrior() {
  std::cout << std::endl << "Singer Trajectory without measurements:" << std::endl;

  int num_knots = 10;
  double dt = 0.1;

  Eigen::Matrix<double, 6, 1> ad; ad << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  Eigen::Matrix<double, 6, 1> qcd; qcd << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
  const auto trajectory = singer::Interface::MakeShared(ad, qcd);
  std::vector<TrajStateVar> traj_state_vars;

  //
  for (int i = 0; i <= num_knots; ++i) {
    Time time(static_cast<double>(i) * dt);
    auto T_k0 = se3::SE3StateVar::MakeShared(lgmath::se3::Transformation());
    auto w_0k_ink = vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());
    auto dw_0k_ink = vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());
    traj_state_vars.emplace_back(time, T_k0, w_0k_ink, dw_0k_ink);
    trajectory->add(time, T_k0, w_0k_ink, dw_0k_ink);
  }

  // Add priors (alternatively, we could lock pose variable)
  lgmath::se3::Transformation init_pose;
  Eigen::Matrix<double, 6, 1> init_velocity = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> init_acceleration; init_acceleration << -1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
  trajectory->addPosePrior(Time(0.0), init_pose, Eigen::Matrix<double, 6, 6>::Identity());
  trajectory->addVelocityPrior(Time(0.0), init_velocity, Eigen::Matrix<double, 6, 6>::Identity());
  trajectory->addAccelerationPrior(Time(0.0), init_acceleration, Eigen::Matrix<double, 6, 6>::Identity());

  //
  using MeasType = Eigen::Matrix<double, 6, 1>;
  std::vector<std::pair<double, MeasType>> measurements;
  auto &meas1 = measurements.emplace_back(0.0, MeasType::Zero());
  meas1.second << -0.067246, -0.035955, -0.081315, -0.172628,  0.017743, -0.040178;
  auto &meas2 = measurements.emplace_back(0.25, MeasType::Zero());
  meas2.second << -0.16302 ,  0.046278, -0.09073 ,  0.005195,  0.072909,  0.012898;
  auto &meas3 = measurements.emplace_back(0.5, MeasType::Zero());
  meas3.second << 0.11394 , -0.123483,  0.040234, -0.068481, -0.08708 , -0.057885;
  auto &meas4 = measurements.emplace_back(0.75, MeasType::Zero());
  meas4.second << -0.031155,  0.005617, -0.116515,  0.090083,  0.046566, -0.153624;
  auto &meas5 = measurements.emplace_back(1.0, MeasType::Zero());
  meas5.second << 0.148825,  0.189589,  0.117878, -0.017992, -0.107075,  0.105445;

  std::vector<BaseCostTerm::Ptr> meas_terms;
  for (const auto& measurement : measurements) {
    Time time(measurement.first);
    const auto w_0k_ink = trajectory->getVelocityInterpolator(time);
    const auto error_func = vspace::vspace_error<6>(w_0k_ink, measurement.second);
    const auto noise_model = StaticNoiseModel<6>::MakeShared(Eigen::Matrix<double, 6, 6>::Identity());
    const auto loss_func = L2LossFunc::MakeShared();
    meas_terms.emplace_back(WeightedLeastSqCostTerm<6>::MakeShared(error_func, noise_model, loss_func));
  }

  // Initialize problem
  steam::OptimizationProblem problem;

  // Add state variables
  for (unsigned int i = 0; i < traj_state_vars.size(); i++) {
    const TrajStateVar& state = traj_state_vars.at(i);
    problem.addStateVariable(state.T_k0);
    problem.addStateVariable(state.w_0k_ink);
    problem.addStateVariable(state.dw_0k_ink);
  }

  // Add cost terms
  trajectory->addPriorCostTerms(problem);
  for (const auto& meas_term : meas_terms)
    problem.addCostTerm(meas_term);

  // Setup Solver and Optimize
  steam::GaussNewtonSolver::Params params;
  params.verbose = true;
  steam::GaussNewtonSolver solver(problem, params);
  solver.optimize();

  // // dump trajectory
  // const double start_time = traj_state_vars.front().time.seconds();
  // const double end_time = traj_state_vars.back().time.seconds();
  // for (double t = start_time; t <= end_time; t += 0.01) {
  //   const auto T_0k_vec = trajectory->getPoseInterpolator(Time(t))->value().inverse().vec();
  //   const auto w_0k_ink = trajectory->getVelocityInterpolator(Time(t))->value();
  //   std::cout << t << " " << T_0k_vec.transpose() << " " << w_0k_ink.transpose() << std::endl;
  // }
}

int main(int argc, char** argv) {
  WNOAPrior();
  WNOJPrior();
  SingerPrior();
  return 0;
}
