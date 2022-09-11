/**
 * \file RatialVelMeasWithConstVelTraj.cpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 * \brief Simple example that uses pointwise radial velocity measurements to
 * estimate body velocity of the vehicle, taking into account motion distortion.
 */
#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace steam;

// clang-format off
int main(int argc, char **argv) {
  // Time horizon, consider this the time for getting a full lidar scan
  const double T = 1.0;

  // The vehicle-inertial transformation at t=0, assuming to be identity for
  // simplicity
  Eigen::Matrix4d T_iv = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d T_vi = T_iv.inverse();

  // The sensor-vehicle transformation, assuming the sensor is 1 meter ahead of
  // the vehicle.
  Eigen::Matrix4d T_vs = Eigen::Matrix4d::Identity();
  T_vs(0, 3) = 1;
  Eigen::Matrix4d T_sv = T_vs.inverse();

  // The ground truth body-velocity of the vehicle - the vehicle is moving
  // forward (x-axis) while rotating (z-axis)
  Eigen::Matrix<double, 6, 1> w_iv_inv;
  w_iv_inv << -2.0, 0.0, 0.0, 0.0, 0.0, 0.8;
  Eigen::Matrix<double, 6, 1> w_is_ins = lgmath::se3::tranAd(T_sv) * w_iv_inv;

  // measurement time
  std::vector<double> ts{0.25 * T, 0.5 * T, 0.75 * T};

  // The homogeneous coordinates of the landmarks in the inertial frame with
  // timestamp being measured
  std::vector<Eigen::Vector4d> lm_ini;
  lm_ini.emplace_back(Eigen::Vector4d{0.0, 2.0, 0.0, 1.0});
  lm_ini.emplace_back(Eigen::Vector4d{2.0, 0.0, 0.0, 1.0});
  lm_ini.emplace_back(Eigen::Vector4d{0.0, -2.0, 0.0, 1.0});

  std::vector<Eigen::Vector4d> lm_ins;
  for (size_t i=0; i< lm_ini.size(); i++) {
    const auto &lm = lm_ini[i];
    const auto &t = ts[i];
    lm_ins.emplace_back(T_sv * lgmath::se3::vec2tran(t * w_iv_inv) * T_vi * lm);
  }

  std::vector<Eigen::Vector4d> dot_lm_ins;
  for (const auto &lm : lm_ins)
    dot_lm_ins.emplace_back(lgmath::se3::point2fs(lm.head<3>(), lm(3)) * w_is_ins);

  Eigen::Matrix<double, 3, 4> D = Eigen::Matrix<double, 3, 4>::Zero();
  D.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  std::vector<double> rv_measurements;
  for (size_t i = 0; i < dot_lm_ins.size(); ++i) {
    const auto &dot_lm = dot_lm_ins.at(i);
    const auto &lm = lm_ins.at(i);
    const double numerator = lm.transpose() * D.transpose() * D * dot_lm;
    const double sq_denomenator = lm.transpose() * D.transpose() * D * lm;
    const double rv = numerator / std::sqrt(sq_denomenator);
    rv_measurements.emplace_back(rv);
  }

  // Initialize problem
  OptimizationProblem problem;

  /// state variables
  // sensor-vehicle transformation - this is fixed
  const auto T_sv_var = se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(T_sv));
  T_sv_var->locked() = true;
  // vehicle pose at t=0, fixed
  const auto T_vi_at0_var = se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(T_vi));
  T_vi_at0_var->locked() = true;
  // vehicle velocity at t=0, to be estimated
  const auto w_iv_inv_at0_var = vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());
  // vehicle pose at t=T, to be estimated
  const auto T_vi_atT_var = se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(T_vi));
  // vehicle velocity at t=0, to be estimated
  const auto w_iv_inv_atT_var = vspace::VSpaceStateVar<6>::MakeShared(Eigen::Matrix<double, 6, 1>::Zero());

  // add states
  problem.addStateVariable(w_iv_inv_at0_var);
  problem.addStateVariable(T_vi_atT_var);
  problem.addStateVariable(w_iv_inv_atT_var);

  // trajectory interface
  Eigen::Matrix<double, 6, 1> Qc_diag = Eigen::Matrix<double, 6, 1>::Ones();
  Qc_diag.diagonal() << 1e0, 1e0, 1e0, 1e0, 1e0, 1e0;
  traj::const_vel::Interface traj(Qc_diag);
  traj.add(traj::Time(0.0), T_vi_at0_var, w_iv_inv_at0_var);
  traj.add(traj::Time(T), T_vi_atT_var, w_iv_inv_atT_var);

  const auto loss_function = L2LossFunc::MakeShared();
  Eigen::Matrix<double, 1, 1> meas_cov = Eigen::Matrix<double, 1, 1>::Identity();
  const auto meas_noise_model = StaticNoiseModel<1>::MakeShared(meas_cov);
  for (size_t i = 0; i < rv_measurements.size(); ++i) {
    const auto &lm = lm_ins.at(i);
    const auto &rv = rv_measurements.at(i);
    const auto w_iv_inv_att_eval = traj.getVelocityInterpolator(traj::Time(ts[i]));
    const auto w_is_ins_eval = se3::compose_velocity(T_sv_var, w_iv_inv_att_eval);
    const auto rv_error = p2p::RadialVelErrorEvaluator::MakeShared(w_is_ins_eval, lm.head<3>(), rv);
    const auto cost_term = WeightedLeastSqCostTerm<1>::MakeShared(rv_error, meas_noise_model, loss_function);
    problem.addCostTerm(cost_term);
  }

  Eigen::Matrix<double, 6, 6> prior_cov = Eigen::Matrix<double, 6, 6>::Zero();
  prior_cov.diagonal() << 1e4, 1e-2, 1e-2, 1e-2, 1e-2, 1e4;
  Eigen::Matrix<double, 6, 1> prior_w_iv_inv_at0 = Eigen::Matrix<double, 6, 1>::Zero();
  traj.addVelocityPrior(traj::Time(0.0), prior_w_iv_inv_at0, prior_cov);

  // add prior cost terms
  traj.addPriorCostTerms(problem);

  // Make solver
  GaussNewtonSolver::Params params;
  params.verbose = true;
  GaussNewtonSolver solver(problem, params);

  // Optimize
  solver.optimize();

  std::cout << "Ground truth body velocity: " << w_iv_inv.transpose() << std::endl;
  std::cout << "Estimated body velocity at t=0: " << w_iv_inv_at0_var->value().transpose() << std::endl;
  std::cout << "Estimated body velocity at t=T: " << w_iv_inv_atT_var->value().transpose() << std::endl;

  return 0;
}