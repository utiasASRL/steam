#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <vector>

using namespace steam;



int main(int argc, char** argv) {

    const unsigned rollout_window = 5;

    const Eigen::Vector2d V_REF {1.2, 1.10};

    const Eigen::Vector2d V_MAX {1.0, 1.0};
    const Eigen::Vector2d V_MIN {-1.0, -1.0};

    const Eigen::Vector2d ACC_MAX {0.1, 0.2};
    const Eigen::Vector2d ACC_MIN {-0.1, -0.2};

    const Eigen::Matrix<double, 1, 2> A {2.0, 1.0};
    const Eigen::Matrix<double, 1, 1> b {1.0};

    // Setup shared loss functions and noise models for all cost terms
    const auto l1Loss = L1LossFunc::MakeShared();
    const auto l2Loss = L2LossFunc::MakeShared();
    const auto sharedVelNoiseModel = steam::StaticNoiseModel<2>::MakeShared(Eigen::Matrix2d::Identity());


    std::vector<vspace::VSpaceStateVar<2>::Ptr> vel_state_vars;
    vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared(Eigen::Vector2d::Zero())); 
    vel_state_vars.front()->locked() = true;

    for (unsigned i = 0; i < rollout_window; i++) {
        vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared(0.0*Eigen::Vector2d::Random())); 
        std::cout << "Initial velo " << vel_state_vars.back()->value() << std::endl;
    }

    steam::Timer timer;
    for (double weight = 1.0; weight > 5e-5; weight *= 0.8) {

        // Setup the optimization problem
        OptimizationProblem opt_problem;

        // Create STEAM variables
        for (const auto &vel_var : vel_state_vars)
        {
            opt_problem.addStateVariable(vel_var);
            const auto vel_cost_term = WeightedLeastSqCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_REF), sharedVelNoiseModel, l2Loss);
            opt_problem.addCostTerm(vel_cost_term);

            opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_MAX), weight));
            opt_problem.addCostTerm(vspace::LogBarrierCostTerm<1>::MakeShared(
              vspace::vspace_error<1>(vspace::MatrixMultEvaluator<1, 2>::MakeShared(vel_var, A), b)
            , weight));
            opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(vel_var, V_MIN)), weight));
        }

        for (unsigned i = 1; i < vel_state_vars.size(); i++)
        {
          const auto accel_term = vspace::add<2>(vel_state_vars[i], vspace::neg<2>(vel_state_vars[i-1]));
          opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(accel_term, ACC_MAX), weight));
          opt_problem.addCostTerm(vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(accel_term, ACC_MIN)), weight));
        }


        // Solve the optimization problem with GaussNewton solver
        //using SolverType = steam::GaussNewtonSolver; // Old solver, does not have back stepping capability
        //using SolverType = steam::LineSearchGaussNewtonSolver;
        using SolverType = LevMarqGaussNewtonSolver;

        // Initialize solver parameters
        SolverType::Params params;
        params.verbose = false; // Makes the output display for debug when true
        params.max_iterations = 100;
        params.absolute_cost_change_threshold = 1e-3;

        SolverType solver(opt_problem, params);

        double initial_cost = opt_problem.cost();
        // Check the cost, disregard the result if it is unreasonable (i.e if its higher then the initial cost)
        std::cout << "The Initial Solution Cost is:" << initial_cost << std::endl;


        // Solve the optimization problem
        solver.optimize();

        double final_cost = opt_problem.cost();

        std::cout << "The Final Solution Cost is:" << final_cost << std::endl;
        
    }
    std::cout << "Total time: " << timer.milliseconds() << "ms" << std::endl;
    for (const auto &vel_var : vel_state_vars)
    {
        std::cout << "Final velo " << vel_var->value() << std::endl;
    }
   


    return 0;
}
