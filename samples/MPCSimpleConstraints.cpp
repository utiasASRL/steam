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
  

  class LateralErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
  public:
    using Ptr = std::shared_ptr<LateralErrorEvaluator>;
    using ConstPtr = std::shared_ptr<const LateralErrorEvaluator>;

    // using PathPtr = tactic::LocalizationChain::Ptr;
    // using PathIter = pose_graph::PathIterator<tactic::LocalizationChain::Parent>;
    // using InType = Eigen::Matrix<double, 6, 1>;
    using InType = Transformation;
    using OutType = Eigen::Matrix<double, 1, 1>;
    using Segment = std::pair<unsigned, unsigned>;

    static Ptr MakeShared(const Evaluable<InType>::ConstPtr& tf,
                          const Transformation seq_end);

    LateralErrorEvaluator(const Evaluable<InType>::ConstPtr& tf,
                          const Transformation seq_end): tf_{tf}, xi_{se3::tran2vec(tf_)},
                          seq_end_{seq_end} {};

    bool active() const override;
    void getRelatedVarKeys(KeySet& keys) const override;

    OutType value() const override;
    typename steam::Node<OutType>::Ptr forward() const override;
    void backward(const Eigen::MatrixXd& lhs, const typename steam::Node<OutType>::Ptr& node,
                  Jacobians& jacs) const override;

  private:
    /** \brief Transform to vec evaluable */
    const Evaluable<InType>::ConstPtr tf_;
    /** \brief Transform to vec evaluable */
    const Evaluable<Eigen::Matrix<double, 6, 1>>::ConstPtr xi_;


    /** Return sequence id of path*/
    const Transformation seq_start_ {};
    const Transformation seq_end_;


  };

  LateralErrorEvaluator::Ptr path_track_error(const Evaluable<LateralErrorEvaluator::InType>::ConstPtr& tf,
                          const Transformation seq_end);
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

  LateralErrorEvaluator::Ptr LateralErrorEvaluator::MakeShared(const Evaluable<InType>::ConstPtr& xi,
                          const Transformation seq_end) {
    return std::make_shared<LateralErrorEvaluator>(xi, seq_end);
  }


  bool LateralErrorEvaluator::active() const { return xi_->active(); }

  void LateralErrorEvaluator::getRelatedVarKeys(KeySet& keys) const {
    xi_->getRelatedVarKeys(keys);
  }

  LateralErrorEvaluator::OutType LateralErrorEvaluator::value() const {

    const auto &[coc, roc] = CurvatureInfo::fromTransform(
                              seq_start_.inverse() * seq_end_);

    Eigen::Vector4d coc_h {0, 0, 0, 1};
    coc_h.head<3>() = coc;

    coc_h = seq_start_.inverse().matrix() * coc_h;

    const auto value = (tf_->value().r_ba_ina() - coc_h.head<3>()).norm() - fabs(roc);
    OutType m_val;
    m_val(0, 0) = value;
    return m_val;
  }

  typename steam::Node<LateralErrorEvaluator::OutType>::Ptr LateralErrorEvaluator::forward() const {
    const auto child = xi_->forward();


    // std::cout << "MPC Pos: " << tf_->value().r_ba_ina() << std::endl;
    // std::cout << "lie algebra Pos: " << lgmath::so3::vec2jac(child->value().tail<3>()) * child->value().head<3>()<< std::endl;

    const auto curve_info  = CurvatureInfo::fromTransform(seq_start_.inverse() * seq_end_);

    Eigen::Vector4d coc_h {0, 0, 0, 1};
    coc_h.head<3>() = curve_info.center;

    coc_h = seq_start_.inverse().matrix() * coc_h;

    
    // std::cout << "pose1: "  << seq_start_.r_ab_inb()<< std::endl;
    // std::cout << "pose2: " << seq_end_.r_ab_inb()<< std::endl;
    // std::cout << "center: " << coc_h.head<3>()<< std::endl;
    
    const auto position_vec = tf_->value().r_ba_ina() - coc_h.head<3>();
    const auto value = position_vec.norm() - fabs(curve_info.radius);
    OutType m_val;
    m_val(0, 0) = value;
    const auto node = steam::Node<OutType>::MakeShared(m_val);
    node->addChild(child);

    // std::cout << "Error: " << value<< std::endl;

    //In case the localization chain changes between forwards and backwards pass
    //Store the info about the curve used for the error
    const auto curve_node = steam::Node<CurvatureInfo>::MakeShared(curve_info);
    node->addChild(curve_node);

    const auto pos_node = steam::Node<Eigen::Vector3d>::MakeShared(position_vec);
    node->addChild(pos_node);

    return node;
  }

  void LateralErrorEvaluator::backward(const Eigen::MatrixXd& lhs,
                                       const steam::Node<OutType>::Ptr& node,
                                       Jacobians& jacs) const {
    if (xi_->active()) {
      const auto child = std::static_pointer_cast<steam::Node<Eigen::Matrix<double, 6, 1>>>(node->at(0));
      const auto curvature_info = std::static_pointer_cast<steam::Node<CurvatureInfo>>(node->at(1));
      const auto pos_node = std::static_pointer_cast<steam::Node<Eigen::Vector3d>>(node->at(2));
      Eigen::Matrix<double, 1, 6> local_jac = Eigen::Matrix<double, 1, 6>::Zero();
      local_jac.head<3>() = -pos_node->value().transpose() * lgmath::so3::vec2jac(child->value().tail<3>()).transpose() / 
                                            (pos_node->value().norm());

      Eigen::MatrixXd new_lhs = lhs * local_jac;
    //   std::cout << "Jac: " << local_jac << std::endl;

      xi_->backward(new_lhs, child, jacs);
    }
  }

  LateralErrorEvaluator::Ptr path_track_error(const Evaluable<LateralErrorEvaluator::InType>::ConstPtr& tf,
                          const Transformation seq_end) {
                            return LateralErrorEvaluator::MakeShared(tf, seq_end);
                        } 

}
}


int main(int argc, char** argv) {

    const unsigned rollout_window = 10;

    const Eigen::Vector2d V_REF {1.1, 0.0};

    const Eigen::Vector2d V_MAX {1.0, 1.0};
    const Eigen::Vector2d V_MIN {-1.0, -1.0};

    const Eigen::Vector2d ACC_MAX {0.1, 0.2};
    const Eigen::Vector2d ACC_MIN {-0.1, -0.2};

    const double DT = 0.1;
    Eigen::Matrix<double, 6, 2> P_tran;
    P_tran << 1, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 0,
                0, 1;

    const Eigen::Matrix<double, 1, 2> A {2.0, 1.0};
    const Eigen::Matrix<double, 1, 1> b {1.0};

    auto tf_from_global = [](double x, double y, double theta) -> lgmath::se3::Transformation {
        auto rotm = lgmath::so3::vec2rot({0, 0, theta});
        Eigen::Vector3d final_pose {x, y, 0};
        return lgmath::se3::Transformation(rotm, -rotm.transpose() * final_pose);
    };
    
    // Setup shared loss functions and noise models for all cost terms
    const auto l1Loss = L1LossFunc::MakeShared();
    const auto l2Loss = L2LossFunc::MakeShared();
    const auto sharedVelNoiseModel = steam::StaticNoiseModel<2>::MakeShared(Eigen::Matrix2d::Identity());
    const auto pathNoiseModel = steam::StaticNoiseModel<1>::MakeShared(0.1*Eigen::Matrix<double, 1, 1>::Identity());
    const auto finalPoseNoiseModel = steam::StaticNoiseModel<6>::MakeShared(1.0*Eigen::Matrix<double, 6, 6>::Identity());


    std::vector<vspace::VSpaceStateVar<2>::Ptr> vel_state_vars;
    std::vector<Evaluable<lgmath::se3::Transformation>::Ptr> pose_vars;
    vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared(Eigen::Vector2d::Zero())); 
    vel_state_vars.front()->locked() = true;

    auto T_init = se3::SE3StateVar::MakeShared(tf_from_global(0.1, 0.1, 0));
    T_init->locked() = true;
    pose_vars.push_back(T_init);

    auto seq_end = tf_from_global(3, 3, M_PI_2);

    for (unsigned i = 0; i < rollout_window; i++) {
        vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared(0.0*Eigen::Vector2d::Random())); 
        std::cout << "Initial velo " << vel_state_vars.back()->value() << std::endl;
    }

    steam::Timer timer;
    for (double weight = 6e-2; weight > 5e-2; weight *= 0.8) {

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

            opt_problem.addStateVariable(vel_var);
            const auto vel_cost_term = WeightedLeastSqCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_REF), sharedVelNoiseModel, l2Loss);
            opt_problem.addCostTerm(vel_cost_term);

            // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_MAX), weight));
            // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<1>::MakeShared(
            //   vspace::vspace_error<1>(vspace::MatrixMultEvaluator<1, 2>::MakeShared(vel_var, A), b)
            // , weight));
            const auto path_cost = WeightedLeastSqCostTerm<1>::MakeShared(vtr::steam_extension::path_track_error(Tf_acc, seq_end), pathNoiseModel, l2Loss);
            // opt_problem.addCostTerm(path_cost);


            // opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(vel_var, V_MIN)), weight));
        }

        const auto end_pose_cost = WeightedLeastSqCostTerm<6>::MakeShared(se3::se3_error(Tf_acc, seq_end), finalPoseNoiseModel, l2Loss);
        opt_problem.addCostTerm(end_pose_cost);

        for (unsigned i = 1; i < vel_state_vars.size(); i++)
        {
          const auto accel_term = vspace::add<2>(vel_state_vars[i], vspace::neg<2>(vel_state_vars[i-1]));
        //   opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(accel_term, ACC_MAX), weight));
        //   opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(accel_term, ACC_MIN)), weight));
        }


        // Solve the optimization problem with GaussNewton solver
        //using SolverType = steam::GaussNewtonSolver; // Old solver, does not have back stepping capability
        using SolverType = steam::DoglegGaussNewtonSolver;
        //using SolverType = LevMarqGaussNewtonSolver;

        // Initialize solver parameters
        SolverType::Params params;
        params.verbose = true; // Makes the output display for debug when true
        params.max_iterations = 1;
        params.absolute_cost_change_threshold = 1e-3;


        double initial_cost = opt_problem.cost();
        // Check the cost, disregard the result if it is unreasonable (i.e if its higher then the initial cost)
        std::cout << "The Initial Solution Cost is:" << initial_cost << std::endl;


        // Solve the optimization problem
        bool solver_converged = false;
        while (!solver_converged) {
            SolverType solver(opt_problem, params);

            solver.optimize();
            for (const auto &pose_var : pose_vars)
            {
                std::cout << "Final pose " << pose_var->value().r_ab_inb().transpose() << std::endl;
            }
            solver_converged = solver.termination_cause() != SolverBase::TERMINATE_MAX_ITERATIONS;
        }

        double final_cost = opt_problem.cost();

        std::cout << "The Final Solution Cost is:" << final_cost << std::endl;
        
    }
    std::cout << "Total time: " << timer.milliseconds() << "ms" << std::endl;
    for (const auto &vel_var : vel_state_vars)
    {
        std::cout << "Final velo " << vel_var->value() << std::endl;
    }


    std::cout << "Target position" << tf_from_global(3*sin(0.367), 3*(1-cos(0.367)), 0.367);
   


    return 0;
}
