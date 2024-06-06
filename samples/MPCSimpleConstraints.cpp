#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <vector>

using namespace steam;



template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

namespace vtr {
namespace steam_extension {

  using namespace lgmath::se3;
  using namespace steam;

  struct CurvatureInfo
  {
    Eigen::Vector3d center;
    double radius;

    inline double curvature() const {
      return 1 / radius;
    }

    static CurvatureInfo fromTransform(const Transformation &T_12); 
  };

  class PathInterpolator : public Evaluable<Transformation> {
    public:
    using Ptr = std::shared_ptr<PathInterpolator>;
    using ConstPtr = std::shared_ptr<const PathInterpolator>;

    using InType = Transformation;
    using OutType = Transformation;

    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& tf,
                          const Transformation seq_start, const Transformation seq_end);

    PathInterpolator(const Evaluable<InType>::ConstPtr& tf,
                          const Transformation seq_start, Transformation seq_end): tf_{tf}, 
                          seq_start_{seq_start}, seq_end_{seq_end} {};

    bool active() const override;
    void getRelatedVarKeys(KeySet& keys) const override {};

    OutType value() const override;
    typename steam::Node<OutType>::Ptr forward() const override;
    void backward(const Eigen::MatrixXd& lhs, const typename steam::Node<OutType>::Ptr& node,
                  Jacobians& jacs) const override {};

  private:
    /** \brief Transform to vec evaluable */
    const Evaluable<InType>::ConstPtr tf_;
    const se3::PoseInterpolator::Ptr path_;

    const se3::ComposeInverseEvaluator::ConstPtr se3_err_;


    /** Return sequence id of path*/
    const Transformation seq_start_;
    const Transformation seq_end_;
  };

  PathInterpolator::Ptr PathInterpolator::MakeShared(const Evaluable<InType>::ConstPtr& tf,
                          const Transformation seq_start, const Transformation seq_end) {
                            return std::make_shared<PathInterpolator>(tf, seq_start, seq_end);
  }

  bool PathInterpolator::active() const{ return false; }

  PathInterpolator::OutType PathInterpolator::value() const{
    Transformation edge = seq_start_.inverse() * seq_end_;
    const auto &[coc, roc] = CurvatureInfo::fromTransform(edge);
    Eigen::Vector4d coc_h {0, 0, 0, 1};
    coc_h.head<3>() = coc;

    coc_h = seq_start_.inverse().matrix() * coc_h;
    

    const auto interp_ang = acos((tf_->value().r_ab_inb() - coc_h.head<3>()).normalized().dot((seq_start_.r_ab_inb() - coc_h.head<3>()).normalized()));
    const auto interp_full = acos((seq_end_.r_ab_inb() - coc_h.head<3>()).normalized().dot((seq_start_.r_ab_inb() - coc_h.head<3>()).normalized()));
    const double interp = interp_ang / interp_full;
    const auto val  = seq_start_ * Transformation(interp * edge.vec(), 0);
    // std::cout << "Interp: " << interp << std::endl;
    //std::cout << "TF: " << val << std::endl;
    return val;
  }

  typename steam::Node<PathInterpolator::OutType>::Ptr PathInterpolator::forward() const {
    return steam::Node<OutType>::MakeShared(value());
  }
  
  

//   class LateralErrorEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
//   public:
//     using Ptr = std::shared_ptr<LateralErrorEvaluator>;
//     using ConstPtr = std::shared_ptr<const LateralErrorEvaluator>;

//     using InType = Transformation;
//     using OutType = Eigen::Matrix<double, 6, 1>;
//     using Segment = std::pair<unsigned, unsigned>;
//     using Time = traj::Time;

//     static Ptr MakeShared(const Evaluable<InType>::ConstPtr& tf,
//                           const Transformation seq_end);

//     LateralErrorEvaluator(const Evaluable<InType>::ConstPtr& tf,
//                           const Transformation seq_end): tf_{tf}, seq_end_{se3::SE3StateVar::MakeShared(seq_end)},
//                           path_{se3::PoseInterpolator::MakeShared(Time(0.0), seq_start_, Time(0.0), seq_end_, Time(1.0))},
//                           se3_err_{se3::compose_rinv(tf_, path_)} {
//                             seq_start_->locked() = true;
//                             seq_end_->locked() = true;
//                           };

//     bool active() const override;
//     void getRelatedVarKeys(KeySet& keys) const override;

//     OutType value() const override;
//     typename steam::Node<OutType>::Ptr forward() const override;
//     void backward(const Eigen::MatrixXd& lhs, const typename steam::Node<OutType>::Ptr& node,
//                   Jacobians& jacs) const override;

//   private:
//     /** \brief Transform to vec evaluable */
//     const Evaluable<InType>::ConstPtr tf_;
//     const se3::PoseInterpolator::Ptr path_;

//     const se3::ComposeInverseEvaluator::ConstPtr se3_err_;


//     /** Return sequence id of path*/
//     const se3::SE3StateVar::Ptr seq_start_ = se3::SE3StateVar::MakeShared(Transformation());
//     const se3::SE3StateVar::Ptr seq_end_;


//   };

//   LateralErrorEvaluator::Ptr path_track_error(const Evaluable<LateralErrorEvaluator::InType>::ConstPtr& tf,
//                           const Transformation seq_end);
}
}


namespace vtr {
namespace steam_extension {

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

//   LateralErrorEvaluator::Ptr LateralErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& xi,
//                           const Transformation seq_end) {
//     return std::make_shared<LateralErrorEvaluator>(xi, seq_end);
//   }


//   bool LateralErrorEvaluator::active() const { return tf_->active(); }

//   void LateralErrorEvaluator::getRelatedVarKeys(KeySet& keys) const {
//     tf_->getRelatedVarKeys(keys);
//   }

//   LateralErrorEvaluator::OutType LateralErrorEvaluator::value() const {

//     const auto &[coc, roc] = CurvatureInfo::fromTransform(
//                               seq_start_->value().inverse() * seq_end_->value());

//     Eigen::Vector4d coc_h {0, 0, 0, 1};
//     coc_h.head<3>() = coc;

//     coc_h = seq_start_->value().inverse().matrix() * coc_h;

//     const auto angle = (tf_->value().r_ba_ina() - coc_h.head<3>()).dot((seq_start_->value().r_ba_ina() - coc_h.head<3>()));
//     *path_ = se3::PoseInterpolator(Time(angle), seq_start_, Time(0.0), seq_end_, Time(1.0));

//     OutType m_val;
//     m_val(0, 0) = value;
//     return m_val;
//   }

//   typename steam::Node<LateralErrorEvaluator::OutType>::Ptr LateralErrorEvaluator::forward() const {
//     const auto child = xi_->forward();


//     // std::cout << "MPC Pos: " << tf_->value().r_ba_ina() << std::endl;
//     // std::cout << "lie algebra Pos: " << lgmath::so3::vec2jac(child->value().tail<3>()) * child->value().head<3>()<< std::endl;

//     const auto curve_info  = CurvatureInfo::fromTransform(seq_start_.inverse() * seq_end_);

//     Eigen::Vector4d coc_h {0, 0, 0, 1};
//     coc_h.head<3>() = curve_info.center;

//     coc_h = seq_start_.inverse().matrix() * coc_h;

    
//     // std::cout << "pose1: "  << seq_start_.r_ab_inb()<< std::endl;
//     // std::cout << "pose2: " << seq_end_.r_ab_inb()<< std::endl;
//     // std::cout << "center: " << coc_h.head<3>()<< std::endl;
    
//     const auto position_vec = tf_->value().r_ba_ina() - coc_h.head<3>();
//     const auto value = position_vec.norm() - fabs(curve_info.radius);
//     OutType m_val;
//     m_val(0, 0) = value;
//     const auto node = steam::Node<OutType>::MakeShared(m_val);
//     node->addChild(child);

//     // std::cout << "Error: " << value<< std::endl;

//     //In case the localization chain changes between forwards and backwards pass
//     //Store the info about the curve used for the error
//     const auto curve_node = steam::Node<CurvatureInfo>::MakeShared(curve_info);
//     node->addChild(curve_node);

//     const auto pos_node = steam::Node<Eigen::Vector3d>::MakeShared(position_vec);
//     node->addChild(pos_node);

//     return node;
//   }

//   void LateralErrorEvaluator::backward(const Eigen::MatrixXd& lhs,
//                                        const steam::Node<OutType>::Ptr& node,
//                                        Jacobians& jacs) const {
//     if (xi_->active()) {
//       const auto child = std::static_pointer_cast<steam::Node<Eigen::Matrix<double, 6, 1>>>(node->at(0));
//       const auto curvature_info = std::static_pointer_cast<steam::Node<CurvatureInfo>>(node->at(1));
//       const auto pos_node = std::static_pointer_cast<steam::Node<Eigen::Vector3d>>(node->at(2));
//       Eigen::Matrix<double, 1, 6> local_jac = Eigen::Matrix<double, 1, 6>::Zero();
//       local_jac.head<3>() = -pos_node->value().transpose() * lgmath::so3::vec2jac(child->value().tail<3>()).transpose() / 
//                                             (pos_node->value().norm());

//       Eigen::MatrixXd new_lhs = lhs * local_jac;
//     //   std::cout << "Jac: " << local_jac << std::endl;

//       xi_->backward(new_lhs, child, jacs);
//     }
//   }

//   LateralErrorEvaluator::Ptr path_track_error(const Evaluable<LateralErrorEvaluator::InType>::ConstPtr& tf,
//                           const Transformation seq_end) {
//                             return LateralErrorEvaluator::MakeShared(tf, seq_end);
//                         } 

}
}


int main(int argc, char** argv) {

    const unsigned rollout_window = 20;

    const Eigen::Vector2d V_REF {1.0, 0.0};
    const Eigen::Vector2d V_INIT {0.1, 0.0};


    const Eigen::Vector2d V_MAX {1.0, 1.0};
    const Eigen::Vector2d V_MIN {-1.0, -1.0};

    const Eigen::Vector2d ACC_MAX {5.0, 10.0};
    const Eigen::Vector2d ACC_MIN {-5.0, -10.0};

    const double DT = 0.1;
    Eigen::Matrix<double, 6, 2> P_tran;
    P_tran << 1, 0,
              0, 0,
              0, 0,
              0, 0,
              0, 0,
              0, 1;

    auto tf_from_global = [](double x, double y, double theta) -> lgmath::se3::Transformation {
        auto rotm = lgmath::so3::vec2rot({0, 0, theta});
        Eigen::Vector3d final_pose {x, y, 0};
        return lgmath::se3::Transformation(rotm, -rotm.transpose() * final_pose);
    };
    
    // Setup shared loss functions and noise models for all cost terms
    const auto l2Loss = L2LossFunc::MakeShared();
    const auto sharedVelNoiseModel = steam::StaticNoiseModel<2>::MakeShared(1.0*Eigen::Matrix2d::Identity());

    Eigen::Matrix<double, 6, 6> pose_cov;
    pose_cov.diagonal() << 1.0, 1.0, 10000000.0, 10000000.0, 10000000.0, 1.0;
    const auto finalPoseNoiseModel = steam::StaticNoiseModel<6>::MakeShared(0.05*pose_cov);

    std::vector<vspace::VSpaceStateVar<2>::Ptr> vel_state_vars;
    vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared(Eigen::Vector2d::Zero())); 
    vel_state_vars.front()->locked() = true;

    auto T_init = se3::SE3StateVar::MakeShared(tf_from_global(0.1, 0.1, 0));
    T_init->locked() = true;

    auto seq_end = tf_from_global(3, -3, -M_PI_2);
    auto T_final = se3::SE3StateVar::MakeShared(seq_end);
    T_final->locked() = true;

    for (unsigned i = 0; i < rollout_window; i++) {
        vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared((i+1) * V_INIT)); 
        std::cout << "Initial velo " << vel_state_vars.back()->value() << std::endl;
    }

    steam::Timer timer;
    for (double weight = 1.0; weight > 5e-2; weight *= 0.9) {

      std::vector<Evaluable<lgmath::se3::Transformation>::Ptr> pose_vars;
      std::vector<Evaluable<lgmath::se3::Transformation>::Ptr> path_vars;

      std::cout << "Weight is: " << weight;

      // Setup the optimization problem
      OptimizationProblem opt_problem;

      Evaluable<lgmath::se3::Transformation>::Ptr Tf_acc = T_init;

      // Create STEAM variables
      for (const auto &vel_var : vel_state_vars)
      {
        auto vel_proj = vspace::MatrixMultEvaluator<6,2>::MakeShared(vel_var, DT*P_tran);
        auto deltaTf = se3::ExpMapEvaluator::MakeShared(vel_proj);
        Tf_acc = se3::compose(Tf_acc, deltaTf);
        pose_vars.push_back(Tf_acc);

        auto path_interp = vtr::steam_extension::PathInterpolator::MakeShared(Tf_acc, lgmath::se3::Transformation(), seq_end);
        path_vars.push_back(path_interp);

        opt_problem.addStateVariable(vel_var);
        const auto vel_cost_term = WeightedLeastSqCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_REF), sharedVelNoiseModel, l2Loss);
        // opt_problem.addCostTerm(vel_cost_term);

        // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_MAX), weight));
        const auto path_cost = WeightedLeastSqCostTerm<6>::MakeShared(se3::tran2vec(se3::compose_rinv(Tf_acc, path_interp)), finalPoseNoiseModel, l2Loss);
        opt_problem.addCostTerm(path_cost);

        // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(vel_var, V_MIN)), weight));
      }

      for (unsigned i = 1; i < vel_state_vars.size(); i++)
      {
        const auto accel_term = vspace::add<2>(vel_state_vars[i], vspace::neg<2>(vel_state_vars[i-1]));
        // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(accel_term, DT*ACC_MAX), weight));
        // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(accel_term, DT*ACC_MIN)), weight));
      }


      // Solve the optimization problem with GaussNewton solver
      //using SolverType = steam::GaussNewtonSolver; // Old solver, does not have back stepping capability
      using SolverType = steam::LineSearchGaussNewtonSolver;
      //using SolverType = LevMarqGaussNewtonSolver;

      // Initialize solver parameters
      SolverType::Params params;
      params.verbose = true; // Makes the output display for debug when true
      params.max_iterations = 100;
      params.absolute_cost_change_threshold = 1e-5;
      params.backtrack_multiplier = 0.75;
    /// Maximimum number of times to backtrack before giving up
      params.max_backtrack_steps = 50;


      double initial_cost = opt_problem.cost();
      // Check the cost, disregard the result if it is unreasonable (i.e if its higher then the initial cost)
      std::cout << "The Initial Solution Cost is:" << initial_cost << std::endl;


      SolverType solver(opt_problem, params);

      try{
        solver.optimize();
      } catch (steam::unsuccessful_step& e) {
        for (const auto &pose_var : pose_vars)
        {
            std::cout << pose_var->value().r_ab_inb().transpose().head<2>() << std::endl;
        }
        std::cout << std::endl;
        for (const auto &pose_var : path_vars)
        {
            std::cout << pose_var->value().r_ab_inb().transpose().head<2>() << std::endl;
        }

        double final_cost = opt_problem.cost();
        std::cout << "The Final Solution Cost is:" << final_cost << std::endl;
        std::cerr << "Steam failed. reporting best guess";

        break;
      }
      
      if (solver.curr_iteration() == 1) {
        std::cout << "Breaking out, no further improvement! \n";
        for (const auto &pose_var : pose_vars)
        {
            std::cout << pose_var->value().r_ab_inb().transpose().head<2>() << std::endl;
        }
        std::cout << std::endl;
        for (const auto &pose_var : path_vars)
        {
            std::cout << pose_var->value().r_ab_inb().transpose().head<2>() << std::endl;
        }

        double final_cost = opt_problem.cost();
        std::cout << "The Final Solution Cost is:" << final_cost << std::endl;
        break;
      }
        
    }
    std::cout << "Total time: " << timer.milliseconds() << "ms" << std::endl;
    for (const auto &vel_var : vel_state_vars)
    {
        std::cout << "Final velo " << vel_var->value() << std::endl;
    }
   


    return 0;
}
