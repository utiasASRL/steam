//
// Created by alec on 07/06/24.
//
#include <lgmath.hpp>
#include <steam.hpp>
using namespace steam;
using namespace steam::traj;

using Transformation = lgmath::se3::Transformation;

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

struct CurvatureInfo {
  Eigen::Vector3d center;
  double radius;

  double curvature() const { return 1 / radius; }

  static CurvatureInfo fromTransform(const Transformation& T_12);
};

CurvatureInfo CurvatureInfo::fromTransform(const Transformation &T) {
  //Note that this is only along a relative path with an origin at 0,0
  //Using the base tf is still required to move into the world frame
  auto aang = lgmath::so3::rot2vec(T.inverse().C_ba());
  double roc = T.r_ba_ina().norm() / 2 / (sin(aang(2) / 2) + 1e-6);

  static Eigen::Matrix3d rotm_perp;
  rotm_perp << 0.0, -1.0, 0.0,
               1.0, 0.0, 0.0,
               0.0, 0.0, 1.0;

  auto dist = T.r_ba_ina().norm();
  auto lin_dir = T.r_ba_ina() / dist;

  Eigen::Vector3d coc = T.r_ba_ina() / 2 + sqrt(roc*roc - dist*dist / 4) * sgn(roc) * rotm_perp * lin_dir;
  return {coc, roc};
}

/** \brief Structure to store trajectory state variables */
struct TrajStateVar {
  Time time;
  se3::SE3StateVar::Ptr pose;
  vspace::VSpaceStateVar<6>::Ptr velocity;
};

lgmath::se3::Transformation tf_from_global(double x, double y, double theta) {
  auto rotm = lgmath::so3::vec2rot({0, 0, theta});
  Eigen::Vector3d final_pose{x, y, 0};
  return lgmath::se3::Transformation(rotm, -rotm.transpose() * final_pose);
}

class PathInterpolator : public Evaluable<Transformation> {
 public:
  using Ptr = std::shared_ptr<PathInterpolator>;
  using ConstPtr = std::shared_ptr<const PathInterpolator>;

  using InType = Transformation;
  using OutType = Transformation;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& tf,
                        const Transformation seq_start,
                        const Transformation seq_end);

  PathInterpolator(const Evaluable<InType>::ConstPtr& tf,
                   const Transformation seq_start, Transformation seq_end)
      : tf_{tf}, seq_start_{seq_start}, seq_end_{seq_end} {};

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override{};

  OutType value() const override;
  typename steam::Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs,
                const typename steam::Node<OutType>::Ptr& node,
                Jacobians& jacs) const override{};

 private:
  /** \brief Transform to vec evaluable */
  const Evaluable<InType>::ConstPtr tf_;
  const se3::PoseInterpolator::Ptr path_;

  const se3::ComposeInverseEvaluator::ConstPtr se3_err_;

  /** Return sequence id of path*/
  const Transformation seq_start_;
  const Transformation seq_end_;
};

PathInterpolator::Ptr PathInterpolator::MakeShared(
    const Evaluable<InType>::ConstPtr& tf, const Transformation seq_start,
    const Transformation seq_end) {
  return std::make_shared<PathInterpolator>(tf, seq_start, seq_end);
}

bool PathInterpolator::active() const { return false; }

PathInterpolator::OutType PathInterpolator::value() const {
  Transformation edge = seq_start_.inverse() * seq_end_;
  const auto& [coc, roc] = CurvatureInfo::fromTransform(edge);
  Eigen::Vector4d coc_h{0, 0, 0, 1};
  coc_h.head<3>() = coc;

  coc_h = seq_start_.inverse().matrix() * coc_h;

  const auto interp_ang =
      acos((tf_->value().r_ab_inb() - coc_h.head<3>())
               .normalized()
               .dot((seq_start_.r_ab_inb() - coc_h.head<3>()).normalized()));
  const auto interp_full =
      acos((seq_end_.r_ab_inb() - coc_h.head<3>())
               .normalized()
               .dot((seq_start_.r_ab_inb() - coc_h.head<3>()).normalized()));
  const double interp = interp_ang / interp_full;
  const auto val = seq_start_ * Transformation(interp * edge.vec(), 0);
  // std::cout << "Interp: " << interp << std::endl;
  // std::cout << "TF: " << val << std::endl;
  return val;
}

typename steam::Node<PathInterpolator::OutType>::Ptr PathInterpolator::forward()
    const {
  return steam::Node<OutType>::MakeShared(value());
}

int main(int argc, char** argv) {
  constexpr unsigned rollout_window = 10;

  Eigen::Matrix<double, 6, 1> V_REF;
  V_REF << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  const Eigen::Vector2d V_INIT{0.75, 0};
  const double DT = 0.1;
  Eigen::Matrix<double, 6, 2> P_tran;
  P_tran << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;

  const Eigen::Vector2d V_MAX{1.0, 1.0};
  const Eigen::Vector2d V_MIN{-1.0, -1.0};

  const Eigen::Vector2d ACC_MAX{5.0, 10.0};
  const Eigen::Vector2d ACC_MIN{-5.0, -10.0};

  // Smoothing factor diagonal
  Eigen::Array<double, 1, 6> Qc_diag;
  Qc_diag << 100.0, 1.0, 1.0, 1.0, 1.0, 100.0;

  const auto l2Loss = L2LossFunc::MakeShared();
  Eigen::Matrix<double, 6, 6> vel_cov;
  vel_cov.diagonal() << 1.0, 0.001, 0.001, 0.001, 0.001, 10.0;
  const auto velNoiseModel = steam::StaticNoiseModel<6>::MakeShared(vel_cov);

  Eigen::Matrix<double, 6, 6> pose_cov;
  pose_cov.diagonal() << 1.0, 1.0, 1e4, 1e4, 1e4, 1.0;
  const auto poseNoiseModel = steam::StaticNoiseModel<6>::MakeShared(1.0*pose_cov);


  // Steam state variables
  std::vector<TrajStateVar> states;
  std::vector<Evaluable<Transformation>::Ptr> path_vars;


  TrajStateVar start;
  start.time = Time(0.0);
  start.pose = se3::SE3StateVar::MakeShared(tf_from_global(0.1, 0.1, 0));
  start.pose->locked() = true;
  start.velocity = vspace::VSpaceStateVar<6>::MakeShared(V_REF);
  start.velocity->locked() = true;
  states.push_back(start);


  auto seq_start = tf_from_global(0, 0, 0);
  auto seq_end = tf_from_global(3, 3, M_PI_2);

  // Setup state variables - initialized at identity / zero
  for (unsigned int i = 1; i < rollout_window; i++) {
    TrajStateVar temp;
    temp.time = Time(i * DT);
    temp.pose = se3::SE3StateVar::MakeShared(
        Transformation(i * DT * V_REF, 0));
    temp.velocity = vspace::VSpaceStateVar<6>::MakeShared(V_REF);
    states.push_back(temp);
  }


  // Initialize problem
  steam::OptimizationProblem problem;

  // Setup Trajectory
  const_vel::Interface traj(0.1*Qc_diag);
  for (const auto& state : states) {
    traj.add(state.time, state.pose, state.velocity);
    auto path_interp = PathInterpolator::MakeShared(state.pose, seq_start, seq_end);
    path_vars.push_back(path_interp);
    const auto path_cost = WeightedLeastSqCostTerm<6>::MakeShared(se3::tran2vec(se3::compose_rinv(state.pose, path_interp)), poseNoiseModel, l2Loss);
    problem.addCostTerm(path_cost);
    const auto vel_cost_term = WeightedLeastSqCostTerm<6>::MakeShared(vspace::vspace_error<6>(state.velocity, V_REF), velNoiseModel, l2Loss);
    problem.addCostTerm(vel_cost_term);
  }

  // Add state variables
  for (const auto& state : states) {
    problem.addStateVariable(state.pose);
    problem.addStateVariable(state.velocity);
  }

  // Add cost terms
  traj.addPriorCostTerms(problem);

  for (unsigned i = 1; i < states.size(); i++)
  {
    const auto accel_term = vspace::add<6>(states[i].velocity, vspace::neg<6>(states[i-1].velocity));

    const auto acc_cost_term = WeightedLeastSqCostTerm<6>::MakeShared(accel_term, velNoiseModel, l2Loss);
    // problem.addCostTerm(acc_cost_term); // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(accel_term, DT*ACC_MAX), weight));
    // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(accel_term, DT*ACC_MIN)), weight));
  }

  ///
  /// Setup Solver and Optimize
  ///
  steam::LevMarqGaussNewtonSolver::Params params;
  params.verbose = true;
  params.ratio_threshold = 0.1;
  steam::LevMarqGaussNewtonSolver solver(problem, params);

  // Optimize
  solver.optimize();

  for (const auto& state : states) {
    std::cout << state.pose->value().r_ab_inb().transpose().head<2>()
              << std::endl;
  }
  std::cout << std::endl;
  for (const auto &pose_var : path_vars) {
    std::cout << pose_var->value().r_ab_inb().transpose().head<2>() << std::endl;
  }

  std::cout << "Kinematically feasible rollout:" << std::endl;

  Transformation kf_acc = states[0].pose->value();
  Eigen::Array<double, 6, 1> kf_mask;
  kf_mask << 1, 0, 0, 0, 0, 1;
  for (const auto& state : states) {
    std::cout << kf_acc.r_ab_inb().transpose().head<2>()
              << std::endl;
    Eigen::Matrix<double, 6, 1> masked_velo = kf_mask * state.velocity->value().array();
    kf_acc *= Transformation(masked_velo*DT, 0);

  }
  std::cout << "Final velos:" << std::endl;
  for (const auto& state : states) {
    std::cout << state.velocity->value().transpose() << std::endl;
  }


  // // Setup shared loss functions and noise models for all cost terms
  // const auto l2Loss = L2LossFunc::MakeShared();
  // const auto sharedVelNoiseModel =
  // steam::StaticNoiseModel<2>::MakeShared(1.0*Eigen::Matrix2d::Identity());
  //
  // Eigen::Matrix<double, 6, 6> pose_cov;
  // pose_cov.diagonal() << 100000.0, 100000.0, 10000000.0, 10000000.0,
  // 10000000.0, 1.0; const auto finalPoseNoiseModel =
  // steam::StaticNoiseModel<6>::MakeShared(pose_cov);
  //
  // std::vector<vspace::VSpaceStateVar<2>::Ptr> vel_state_vars;
  // vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared(Eigen::Vector2d::Zero()));
  // vel_state_vars.front()->locked() = true;
  //
  // auto T_init = se3::SE3StateVar::MakeShared(tf_from_global(0.1, 0.1, 0));
  // T_init->locked() = true;
  //
  // auto seq_end = tf_from_global(3, 3, M_PI_2);
  // auto T_final = se3::SE3StateVar::MakeShared(seq_end);
  // T_final->locked() = true;
  //
  // for (unsigned i = 0; i < rollout_window; i++) {
  //     vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared((i+1) *
  //     V_INIT)); std::cout << "Initial velo " <<
  //     vel_state_vars.back()->value() << std::endl;
  // }
  //
  // steam::Timer timer;
  // for (double weight = 1.0; weight > 5e-2; weight *= 0.9) {
  //
  //   std::vector<Evaluable<lgmath::se3::Transformation>::Ptr> pose_vars;
  //   std::vector<Evaluable<lgmath::se3::Transformation>::Ptr> path_vars;
  //
  //     std::cout << "Weight is: " << weight;
  //
  //     // Setup the optimization problem
  //     OptimizationProblem opt_problem;
  //
  //     Evaluable<lgmath::se3::Transformation>::Ptr Tf_acc = T_init;
  //
  //     // Create STEAM variables
  //     for (const auto &vel_var : vel_state_vars)
  //     {
  //       auto vel_proj = vspace::MatrixMultEvaluator<6,2>::MakeShared(vel_var,
  //       DT*P_tran); auto deltaTf =
  //       se3::ExpMapEvaluator::MakeShared(vel_proj); Tf_acc =
  //       se3::compose(vtr::steam_extension::block_grad<lgmath::se3::Transformation>(Tf_acc),
  //       deltaTf); pose_vars.push_back(Tf_acc);
  //
  //       auto path_interp =
  //       vtr::steam_extension::PathInterpolator::MakeShared(Tf_acc,
  //       lgmath::se3::Transformation(), seq_end);
  //       path_vars.push_back(path_interp);
  //
  //       opt_problem.addStateVariable(vel_var);
  //       const auto vel_cost_term =
  //       WeightedLeastSqCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var,
  //       V_REF), sharedVelNoiseModel, l2Loss);
  //       opt_problem.addCostTerm(vel_cost_term);
  //
  //       //
  //       opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var,
  //       V_MAX), weight)); const auto path_cost =
  //       WeightedLeastSqCostTerm<6>::MakeShared(se3::tran2vec(se3::compose_rinv(Tf_acc,
  //       path_interp)), finalPoseNoiseModel, l2Loss);
  //       opt_problem.addCostTerm(path_cost);
  //
  //       //
  //       opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(vel_var,
  //       V_MIN)), weight));
  //     }
  //
  //     for (unsigned i = 1; i < vel_state_vars.size(); i++)
  //     {
  //       const auto accel_term = vspace::add<2>(vel_state_vars[i],
  //       vspace::neg<2>(vel_state_vars[i-1]));
  //       //
  //       opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(accel_term,
  //       DT*ACC_MAX), weight));
  //       //
  //       opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(accel_term,
  //       DT*ACC_MIN)), weight));
  //     }
  //
  //
  //     // Solve the optimization problem with GaussNewton solver
  //     //using SolverType = steam::GaussNewtonSolver; // Old solver, does not
  //     have back stepping capability using SolverType =
  //     steam::LineSearchGaussNewtonSolver;
  //     //using SolverType = LevMarqGaussNewtonSolver;
  //
  //     // Initialize solver parameters
  //     SolverType::Params params;
  //     params.verbose = true; // Makes the output display for debug when true
  //     params.max_iterations = 100;
  //     params.absolute_cost_change_threshold = 1e-5;
  //     params.backtrack_multiplier = 0.75;
  //   /// Maximimum number of times to backtrack before giving up
  //     params.max_backtrack_steps = 50;
  //
  //
  //     double initial_cost = opt_problem.cost();
  //     // Check the cost, disregard the result if it is unreasonable (i.e if
  //     its higher then the initial cost) std::cout << "The Initial Solution
  //     Cost is:" << initial_cost << std::endl;
  //
  //
  //     SolverType solver(opt_problem, params);
  //
  //     try{
  //       solver.optimize();
  //     } catch (steam::unsuccessful_step& e) {
  //       for (const auto &pose_var : pose_vars)
  //       {
  //           std::cout << pose_var->value().r_ab_inb().transpose().head<2>()
  //           << std::endl;
  //       }
  //       std::cout << std::endl;
  //       for (const auto &pose_var : path_vars)
  //       {
  //           std::cout << pose_var->value().r_ab_inb().transpose().head<2>()
  //           << std::endl;
  //       }
  //
  //       double final_cost = opt_problem.cost();
  //       std::cout << "The Final Solution Cost is:" << final_cost <<
  //       std::endl; std::cerr << "Steam failed. reporting best guess \n";
  //
  //       return -1;
  //     }
  //
  //     if (solver.curr_iteration() == 1) {
  //       std::cout << "Breaking out, no further improvement! \n";
  //       for (const auto &pose_var : pose_vars)
  //       {
  //           std::cout << pose_var->value().r_ab_inb().transpose().head<2>()
  //           << std::endl;
  //       }
  //       std::cout << std::endl;
  //       for (const auto &pose_var : path_vars)
  //       {
  //           std::cout << pose_var->value().r_ab_inb().transpose().head<2>()
  //           << std::endl;
  //       }
  //
  //       double final_cost = opt_problem.cost();
  //       std::cout << "The Final Solution Cost is:" << final_cost <<
  //       std::endl; break;
  //     }
  //
  //   }
  //   std::cout << "Total time: " << timer.milliseconds() << "ms" << std::endl;
  //

  return 0;
}
