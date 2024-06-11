#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <utility>
#include <vector>

using namespace steam;

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

namespace vtr::steam_extension {

using namespace lgmath::se3;
using namespace steam;

struct CurvatureInfo {
  Eigen::Vector3d center;
  double radius;

  double curvature() const { return 1 / radius; }

  static CurvatureInfo fromTransform(const Transformation& T_12);
};

class PositionEvaluator : public Evaluable<Eigen::Vector3d> {
 public:
  using Ptr = std::shared_ptr<PositionEvaluator>;
  using ConstPtr = std::shared_ptr<const PositionEvaluator>;

  using InType = Transformation;
  using OutType = Eigen::Vector3d;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& in);

  explicit PositionEvaluator(typename Evaluable<InType>::ConstPtr eval)
      : eval_{std::move(eval)} {};

  bool active() const override { return eval_->active(); }
  void getRelatedVarKeys(KeySet& keys) const override {
    eval_->getRelatedVarKeys(keys);
  }

  OutType value() const override { return eval_->value().r_ba_ina(); }
  typename steam::Node<OutType>::Ptr forward() const override {
    return steam::Node<OutType>::MakeShared(value());
  }
  void backward(const Eigen::MatrixXd& lhs,
                const typename steam::Node<OutType>::Ptr& node,
                Jacobians& jacs) const override {
    if (eval_->active()) {
      eval_->backward(lhs, steam::Node<InType>::MakeShared(eval_->value()),
                      jacs);
    }
  };

 private:
  /** \brief Transform to vec evaluable */
  const typename Evaluable<InType>::ConstPtr eval_;
};

PositionEvaluator::Ptr PositionEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& in) {
  return std::make_shared<PositionEvaluator>(in);
}

template <class T>
class GradientBlocker : public Evaluable<T> {
 public:
  using Ptr = std::shared_ptr<GradientBlocker>;
  using ConstPtr = std::shared_ptr<const GradientBlocker>;

  using InType = T;
  using OutType = T;
  using KeySet = typename Evaluable<T>::KeySet;

  static Ptr MakeShared(const typename Evaluable<InType>::ConstPtr& in);

  explicit GradientBlocker(typename Evaluable<InType>::ConstPtr eval)
      : eval_{std::move(eval)} {};

  bool active() const override { return false; }
  void getRelatedVarKeys(KeySet& keys) const override {}

  T value() const override { return eval_->value(); }
  typename steam::Node<T>::Ptr forward() const override {
    return steam::Node<T>::MakeShared(value());
  }
  void backward(const Eigen::MatrixXd& lhs,
                const typename steam::Node<T>::Ptr& node,
                Jacobians& jacs) const override{};

 private:
  /** \brief Transform to vec evaluable */
  const typename Evaluable<T>::ConstPtr eval_;
};

template <class T>
typename GradientBlocker<T>::Ptr GradientBlocker<T>::MakeShared(
    const typename Evaluable<T>::ConstPtr& in) {
  return std::make_shared<GradientBlocker<T>>(in);
}

template <class T>
typename GradientBlocker<T>::Ptr block_grad(
    const typename Evaluable<T>::ConstPtr& in) {
  return GradientBlocker<T>::MakeShared(in);
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
  Node<OutType>::Ptr forward() const override;
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
  Transformation edge = seq_start_ * seq_end_.inverse();
  const auto& [coc, roc] = CurvatureInfo::fromTransform(edge);
  Eigen::Vector4d coc_h{0, 0, 0, 1};
  coc_h.head<3>() = coc;

  coc_h = seq_start_.inverse().matrix() * coc_h;

  const auto interp_ang =
      acos((tf_->value().r_ba_ina() - coc_h.head<3>())
               .normalized()
               .dot((seq_start_.r_ba_ina() - coc_h.head<3>()).normalized()));
  const auto interp_full =
      acos((seq_end_.r_ba_ina() - coc_h.head<3>())
               .normalized()
               .dot((seq_start_.r_ba_ina() - coc_h.head<3>()).normalized()));
  const double interp = interp_ang / interp_full;
  const auto val = Transformation(-interp * edge.vec(), 0) * seq_start_;
  // std::cout << "Interp: " << interp << std::endl;
  // std::cout << "TF: " << val << std::endl;
  return val;
}

typename steam::Node<PathInterpolator::OutType>::Ptr PathInterpolator::forward()
    const {
  return steam::Node<OutType>::MakeShared(value());
}


CurvatureInfo CurvatureInfo::fromTransform(const Transformation& T) {
  // Note that this is only along a relative path with an origin at 0,0
  // Using the base tf is still required to move into the world frame
  auto aang = lgmath::so3::rot2vec(T.C_ba());
  double roc = T.r_ba_ina().norm() / 2 / (sin(aang(2) / 2) + 1e-6);

  static Eigen::Matrix3d rotm_perp;
  rotm_perp << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  auto dist = T.r_ab_inb().norm();
  auto lin_dir = T.r_ab_inb().normalized();;

  Eigen::Vector3d coc = T.r_ab_inb() / 2 + sqrt(roc * roc - dist * dist / 4) *
                                               sgn(roc) * rotm_perp * lin_dir;
  return {coc, roc};
}

}  // namespace vtr::steam_extension

int main(int argc, char** argv) {
  const unsigned rollout_window = 20;

  const Eigen::Vector2d V_REF{1.0, 0.0};
  const Eigen::Vector2d V_INIT{0.45, 0.0};

  const Eigen::Vector2d V_MAX{1.0, 1.0};
  const Eigen::Vector2d V_MIN{-1.0, -1.0};

  const Eigen::Vector2d ACC_MAX{5.0, 3.0};
  const Eigen::Vector2d ACC_MIN{-5.0, -3.0};

  const double DT = 0.1;
  Eigen::Matrix<double, 6, 2> P_tran;
  P_tran << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;


  auto tf_from_global = [](double x, double y,
                           double theta) -> lgmath::se3::Transformation {
    auto rotm = lgmath::so3::vec2rot({0, 0, theta});
    Eigen::Vector3d final_pose{x, y, 0};
    return lgmath::se3::Transformation(rotm, -rotm.transpose() * final_pose).inverse();
  };

  // Setup shared loss functions and noise models for all cost terms
  const auto l2Loss = L2LossFunc::MakeShared();
  const auto sharedVelNoiseModel = steam::StaticNoiseModel<2>::MakeShared(
      10.0 * Eigen::Matrix2d::Identity());

  Eigen::Matrix<double, 6, 6> pose_cov =
      Eigen::Matrix<double, 6, 6>::Identity();
  pose_cov.diagonal() << 1.0, 1.0, 1e4, 1e4, 1e4, 1e3;
  const auto finalPoseNoiseModel =
      steam::StaticNoiseModel<6>::MakeShared(pose_cov);

  Eigen::Matrix4d position_cov = Eigen::Matrix4d::Identity();
  position_cov.diagonal() << 1.0, 1.0, 1.0, 1e4;
  const auto positionNoiseModel =
      steam::StaticNoiseModel<4>::MakeShared(position_cov);

  std::vector<vspace::VSpaceStateVar<2>::Ptr> vel_state_vars;
  vel_state_vars.push_back(
      vspace::VSpaceStateVar<2>::MakeShared(V_INIT));
  // vel_state_vars.front()->locked() = true;

  auto T_init = se3::SE3StateVar::MakeShared(tf_from_global(0.8, 0.2, 0));
  T_init->locked() = true;

  auto seq_start = tf_from_global(0, -2, 0);
  auto seq_end = tf_from_global(3, 1, M_PI_2);

  auto T_final = se3::SE3StateVar::MakeShared(seq_end);
  T_final->locked() = true;

  for (unsigned i = 0; i < rollout_window; i++) {
    vel_state_vars.push_back(
        vspace::VSpaceStateVar<2>::MakeShared(V_INIT));
    std::cout << "Initial velo " << vel_state_vars.back()->value() << std::endl;
  }

  steam::Timer timer;

  std::vector<Evaluable<lgmath::se3::Transformation>::Ptr> pose_vars;
  std::vector<Evaluable<lgmath::se3::Transformation>::Ptr> path_vars;

  for (double weight = 1e-1; weight > 1e-4; weight *= 0.8) {
    pose_vars.clear();
    path_vars.clear();

    std::cout << "Weight is: " << weight;

    // Setup the optimization problem
    OptimizationProblem opt_problem;

    Evaluable<lgmath::se3::Transformation>::Ptr Tf_acc = T_init;

    // Create STEAM variables
    for (const auto& vel_var : vel_state_vars) {
      auto vel_proj =
          vspace::MatrixMultEvaluator<6, 2>::MakeShared(vel_var, DT * P_tran);
      auto deltaTf = se3::ExpMapEvaluator::MakeShared(vel_proj);
      Tf_acc = se3::compose(Tf_acc, deltaTf);
      pose_vars.push_back(Tf_acc);

      // vtr::steam_extension::block_grad<lgmath::se3::Transformation>(

      auto path_interp = vtr::steam_extension::PathInterpolator::MakeShared(
          Tf_acc, seq_start, seq_end);
      const auto interp_state =
          se3::SE3StateVar::MakeShared(path_interp->value().inverse());
      interp_state->locked() = true;
      path_vars.push_back(interp_state);

      opt_problem.addStateVariable(vel_var);
      const auto vel_cost_term = WeightedLeastSqCostTerm<2>::MakeShared(
          vspace::vspace_error<2>(vel_var, V_REF), sharedVelNoiseModel, l2Loss);
      // opt_problem.addCostTerm(vel_cost_term);

      // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var,
      // V_MAX), weight));
      // const auto path_cost = WeightedLeastSqCostTerm<6>::MakeShared(se3::se3_error(Tf_acc,
      //   interp_state->value()), finalPoseNoiseModel, l2Loss);
      const auto path_cost = WeightedLeastSqCostTerm<6>::MakeShared(
                  se3::tran2vec(se3::compose(interp_state, Tf_acc)),
          finalPoseNoiseModel, l2Loss);
      opt_problem.addCostTerm(path_cost);

      // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(vel_var,
      // V_MIN)), weight));
    }

    for (unsigned i = 1; i < vel_state_vars.size(); i++) {
      const auto accel_term = vspace::add<2>(
          vel_state_vars[i], vspace::neg<2>(vel_state_vars[i - 1]));
      // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(accel_term,
      // DT*ACC_MAX), weight));
      // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(accel_term,
      // DT*ACC_MIN)), weight));
    }

    // Solve the optimization problem with GaussNewton solver
    // using SolverType = steam::GaussNewtonSolver; // Old solver, does not have
    // back stepping capability using SolverType =
    // steam::LineSearchGaussNewtonSolver;
    using SolverType = LevMarqGaussNewtonSolver;

    // Initialize solver parameters
    SolverType::Params params;
    params.verbose = true;  // Makes the output display for debug when true
    params.max_iterations = 100;
    params.ratio_threshold = 0.1;
    params.relative_cost_change_threshold = 1e-2;
    //   params.backtrack_multiplier = 0.75;
    // /// Maximimum number of times to backtrack before giving up
    //   params.max_backtrack_steps = 50;

    double initial_cost = opt_problem.cost();
    // Check the cost, disregard the result if it is unreasonable (i.e if its
    // higher then the initial cost)
    std::cout << "The Initial Solution Cost is:" << initial_cost << std::endl;

    SolverType solver(opt_problem, params);

    try {
      solver.optimize();
    } catch (...) {
      for (const auto& pose_var : pose_vars) {
        std::cout << pose_var->value().r_ba_ina().transpose().head<2>()
                  << std::endl;
      }
      std::cout << std::endl;
      for (const auto& pose_var : path_vars) {
        std::cout << pose_var->value().r_ba_ina().transpose().head<2>()
                  << std::endl;
      }

      double final_cost = opt_problem.cost();
      std::cout << "The Final Solution Cost is:" << final_cost << std::endl;
      std::cerr << "Steam failed. reporting best guess \n";
      std::cerr << "Took " << solver.curr_iteration() << " steps \n";

      break;
    }

    if(solver.curr_iteration() == 1) {

      double final_cost = opt_problem.cost();
      std::cout << "The Final Solution Cost is:" << final_cost << std::endl;

      break;
    }
  }
  for (const auto& pose_var : pose_vars) {
    std::cout << pose_var->value().r_ba_ina().transpose().head<2>()
             // << " " << pose_var->value().vec().tail<1>()
              << std::endl;
  }
  std::cout << std::endl;
  for (const auto& pose_var : path_vars) {
    std::cout << pose_var->value().r_ba_ina().transpose().head<2>()
              << std::endl;
  }

  std::cout << "Total time: " << timer.milliseconds() << "ms" << std::endl;
  for (const auto& vel_var : vel_state_vars) {
    std::cout << "Final velo " << vel_var->value() << std::endl;
  }

  return 0;
}
