/**
 * \file LandmarkSLAMExample.cpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace steam;

/** \brief Structure to store trajectory state variables */
struct TrajStateVar {
  traj::Time t;
  se3::SE3StateVar::Ptr T_ri;
  vspace::VSpaceStateVar<6>::Ptr w_ir_inr;
};

/** \brief Example that loads and solves simple bundle adjustment problems */
int main(int argc, char** argv) {
  const int K = 10;
  const int L = 10;

  std::vector<double> t_list;
  std::vector<Eigen::Matrix<double, 6, 1>> gt_w_ir_inr_list;
  std::vector<Eigen::Matrix<double, 4, 4>> gt_T_ri_list;
  std::vector<Eigen::Matrix<double, 6, 1>> w_ir_inr_list;
  std::vector<Eigen::Matrix<double, 4, 4>> T_ri_list;
  for (int i = 0; i < K; ++i) {
    // time (time along the trajectory from start to end)
    t_list.emplace_back(i * L / (K - 1));

    // ground truth trajectory
    if (i == 0) {
      Eigen::Matrix<double, 6, 1> w_ir_inr;
      w_ir_inr << -1., 0., 0., 0., 0., 0.;
      gt_w_ir_inr_list.emplace_back(Eigen::Matrix<double, 6, 1>::Zero());
      gt_T_ri_list.emplace_back(Eigen::Matrix<double, 4, 4>::Identity());
    } else {
      Eigen::Matrix<double, 6, 1> w_ir_inr;
      w_ir_inr << -1., 0., 0., 0., .1, .3;
      gt_w_ir_inr_list.emplace_back(w_ir_inr);
      gt_T_ri_list.emplace_back(
          lgmath::se3::vec2tran(L / (K - 1) * gt_w_ir_inr_list[i - 1]) *
          gt_T_ri_list[i - 1]);
    }

    // initial trajectory
    Eigen::Matrix<double, 6, 1> w_ir_inr;
    w_ir_inr << -1., 0., 0., 0., 0., 0.;
    w_ir_inr_list.emplace_back(w_ir_inr);

    Eigen::Matrix<double, 4, 4> T_ri = Eigen::Matrix<double, 4, 4>::Identity();
    T_ri(0, 3) = -t_list.back();
    T_ri_list.emplace_back(T_ri);
  }

  int Jland = 5;

  std::vector<Eigen::Matrix<double, 4, 1>> gt_p_list;
  gt_p_list.emplace_back(Eigen::Matrix<double, 4, 1>{3., -1., 1., 1.});
  gt_p_list.emplace_back(Eigen::Matrix<double, 4, 1>{0., -1., 2., 1.});
  gt_p_list.emplace_back(Eigen::Matrix<double, 4, 1>{1., -4., 2., 1.});
  gt_p_list.emplace_back(Eigen::Matrix<double, 4, 1>{2., -3., 2., 1.});
  gt_p_list.emplace_back(Eigen::Matrix<double, 4, 1>{1., -3., -2., 1.});

  std::vector<Eigen::Matrix<double, 4, 1>> p_list;
  for (int i = 0; i < Jland; ++i)
    p_list.emplace_back(Eigen::Matrix<double, 4, 1>{0., 0., 0., 1.});

  std::vector<TrajStateVar> traj_state_var_list;
  for (int i = 0; i < K; ++i) {
    traj::Time t(t_list[i]);
    auto w_ir_inr = vspace::VSpaceStateVar<6>::MakeShared(w_ir_inr_list[i]);
    auto T_ri =
        se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(T_ri_list[i]));
    traj_state_var_list.emplace_back(TrajStateVar{t, T_ri, w_ir_inr});
  }
  std::vector<stereo::HomoPointStateVar::Ptr> p_var_list;
  for (int i = 0; i < Jland; ++i)
    p_var_list.emplace_back(
        stereo::HomoPointStateVar::MakeShared(p_list[i].block<3, 1>(0, 0)));

  // create a trajectory
  Eigen::Matrix<double, 6, 6> Qc = Eigen::Matrix<double, 6, 6>::Identity();
  Qc.diagonal() << 0.001, 0.00001, 0.00001, 0.0001, 0.0001, 0.0001;
  traj::const_vel::Interface traj(Qc.inverse());
  for (const auto& state : traj_state_var_list)
    traj.add(state.t, state.T_ri, state.w_ir_inr);

  //
  std::vector<BaseCostTerm::Ptr> cost_term_list;
  const auto loss_func = L2LossFunc::MakeShared();
  Eigen::Matrix3d meas_cov = Eigen::Matrix3d::Identity() * 0.01;
  const auto noise_model = StaticNoiseModel<3>::MakeShared(meas_cov);
  for (int k = 0; k < K; ++k) {
    int j = k % Jland;
    Eigen::Matrix<double, 4, 1> y = gt_T_ri_list[k] * gt_p_list[j];
    const auto error_func = stereo::homo_point_error(
        stereo::compose(traj_state_var_list[k].T_ri, p_var_list[j]), y);
    cost_term_list.emplace_back(WeightedLeastSqCostTerm<3>::MakeShared(
        error_func, noise_model, loss_func));
  }

  //
  traj_state_var_list[0].T_ri->locked() = true;
  traj_state_var_list[0].w_ir_inr->locked() = true;

  //
  OptimizationProblem problem;
  // add state variables
  for (const auto& state : traj_state_var_list) {
    problem.addStateVariable(state.T_ri);
    problem.addStateVariable(state.w_ir_inr);
  }
  // add landmark variables
  for (const auto& p_var : p_var_list) problem.addStateVariable(p_var);
  // add trajectory cost terms
  traj.addPriorCostTerms(problem);
  // add meas cost terms
  for (const auto& cost : cost_term_list) problem.addCostTerm(cost);

  using SolverType = VanillaGaussNewtonSolver;
  SolverType::Params params;
  params.verbose = true;
  SolverType solver(&problem, params);
  solver.optimize();

  return 0;
}