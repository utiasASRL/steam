#include <iostream>

#include <lgmath.hpp>
#include <steam.hpp>
#include <vector>

using namespace steam;

namespace steam {
namespace vspace {


template <int DIM>
class LogBarrierCostTerm : public BaseCostTerm {
 public:
  using Ptr = std::shared_ptr<LogBarrierCostTerm<DIM>>;
  using ConstPtr = std::shared_ptr<const LogBarrierCostTerm<DIM>>;

  using ErrorType = Eigen::Matrix<double, DIM, 1>;  // DIM is measurement dim

  static Ptr MakeShared(
      const typename Evaluable<ErrorType>::ConstPtr &error_function,
      const double weight);
  LogBarrierCostTerm(
      const typename Evaluable<ErrorType>::ConstPtr &error_function,
      const double weight);

  /**
   * \brief Evaluates the cost of this term. Error is first whitened by the
   * noise model and then passed through the loss function, as in:
   *     cost = -log(e)
   */
  double cost() const override;

  /** \brief Get keys of variables related to this cost term */
  void getRelatedVarKeys(KeySet &keys) const override;

  /**
   * \brief Add the contribution of this cost term to the left-hand (Hessian)
   * and right-hand (gradient vector) sides of the Gauss-Newton system of
   * equations.
   */
  void buildGaussNewtonTerms(const StateVector &state_vec,
                             BlockSparseMatrix *approximate_hessian,
                             BlockVector *gradient_vector) const override;

 private:
  /** \brief Error evaluator */
  typename Evaluable<ErrorType>::ConstPtr error_function_;
  /** \brief Noise model */
  double t_ = 1.0;
};

template <int DIM>
auto LogBarrierCostTerm<DIM>::MakeShared(
    const typename Evaluable<ErrorType>::ConstPtr &error_function,
    const double weight) -> Ptr {
  return std::make_shared<LogBarrierCostTerm<DIM>>(
      error_function, weight);
}

template <int DIM>
LogBarrierCostTerm<DIM>::LogBarrierCostTerm(
    const typename Evaluable<ErrorType>::ConstPtr &error_function,
    const double weight)
    : error_function_(error_function),
      t_(weight){
    if ((error_function_->evaluate().array() <= 0).any()) {
      std::cerr << "Error function val: " << error_function_->evaluate();
      throw std::logic_error("value of error is less than 0. Violation of barrier Please init with a feasible point");
    }
  }

template <int DIM>
double LogBarrierCostTerm<DIM>::cost() const {
  return -t_*Eigen::log(error_function_->evaluate().array()).sum();
}

template <int DIM>
void LogBarrierCostTerm<DIM>::getRelatedVarKeys(KeySet &keys) const {
  error_function_->getRelatedVarKeys(keys);
}

template <int DIM>
void LogBarrierCostTerm<DIM>::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {


  Jacobians jacobian_container;
  const auto &jacobians = jacobian_container.get();
  ErrorType error = error_function_->evaluate(Eigen::Matrix<double, DIM, DIM>::Identity(), jacobian_container);

  if ((error.array() <= 0).any()) {
    throw std::logic_error("value of error is less than 0. Violation of barrier");
  }

  const ErrorType inv_err_vec = {1.0 / error.array()};


  // Get map keys into a vector for sorting
  std::vector<StateKey> keys;
  keys.reserve(jacobians.size());
  std::transform(jacobians.begin(), jacobians.end(), std::back_inserter(keys),
                 [](const auto &pair) { return pair.first; });

  // For each jacobian
  for (size_t i = 0; i < keys.size(); i++) {
    const auto &key1 = keys.at(i);
    const auto &jac1 = jacobians.at(key1);

    // std::cout << jac1 << std::endl;


    // Get the key and state range affected
    unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);


    const auto gradTermMatFunc = [&](const Eigen::MatrixXd &jac) -> Eigen::MatrixXd {
      Eigen::MatrixXd newGradTermMat;
      newGradTermMat.resize(jac.cols(), jac.rows());
      newGradTermMat.setZero();
      for (u_int row = 0; row < jac.rows(); ++row) 
      {
        newGradTermMat.col(row) = jac.row(row).transpose() * inv_err_vec(row);
      }
      return newGradTermMat;
    };
    
    Eigen::MatrixXd gradTermMat1 = gradTermMatFunc(jac1);

    Eigen::VectorXd newGradTerm = gradTermMat1.rowwise().sum();

    // Calculate terms needed to update the right-hand-side
    
    // std::cout << "Grad: " << newGradTerm << std::endl;
    // std::cout << "Hess: " << newGradTermMat.transpose() * newGradTermMat << std::endl;


// Update the right-hand side (thread critical)
#pragma omp critical(b_update)
    { gradient_vector->mapAt(blkIdx1) -= -t_ * newGradTerm; }

    // For each jacobian (in upper half)
    for (size_t j = i; j < keys.size(); j++) {
      const auto &key2 = keys.at(j);
      const auto &jac2 = jacobians.at(key2);

      // Get the key and state range affected
      unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

      // Calculate terms needed to update the Gauss-Newton left-hand side
      unsigned int row, col;
      const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
        
        Eigen::MatrixXd gradTermMat2 = gradTermMatFunc(jac2);
        if (blkIdx1 <= blkIdx2) {
          row = blkIdx1;
          col = blkIdx2;

          return t_ * gradTermMat1 * gradTermMat2.transpose();
        } else {
          row = blkIdx2;
          col = blkIdx1;
          return t_ * gradTermMat2 * gradTermMat1.transpose();
        }
      }();

      // Update the left-hand side (thread critical)
      BlockSparseMatrix::BlockRowEntry &entry =
          approximate_hessian->rowEntryAt(row, col, true);
      omp_set_lock(&entry.lock);
      entry.data += newHessianTerm;
      omp_unset_lock(&entry.lock);

    }  // end row loop
  }    // end column loop
}


}  // namespace vspace
}  // namespace steam



int main(int argc, char** argv) {

    const unsigned rollout_window = 5;

    const Eigen::Vector2d V_REF {1.2, 1.10};

    const Eigen::Vector2d V_MAX {1.0, 1.0};
    const Eigen::Vector2d V_MIN {-1.0, -1.0};

    const Eigen::Matrix<double, 1, 2> A {1.0, 1.0};
    const Eigen::Matrix<double, 1, 1> b {1.0};

    // Setup shared loss functions and noise models for all cost terms
    const auto l1Loss = L1LossFunc::MakeShared();
    const auto l2Loss = L2LossFunc::MakeShared();
    const auto sharedVelNoiseModel = steam::StaticNoiseModel<2>::MakeShared(Eigen::Matrix2d::Identity());


    std::vector<vspace::VSpaceStateVar<2>::Ptr> vel_state_vars;
    for (unsigned i = 0; i < rollout_window; i++) {
        vel_state_vars.push_back(vspace::VSpaceStateVar<2>::MakeShared(0.5*Eigen::Vector2d::Random())); 
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
            const auto vel_cost_term = steam::WeightedLeastSqCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_REF), sharedVelNoiseModel, l2Loss);
            opt_problem.addCostTerm(vel_cost_term);

            opt_problem.addCostTerm(steam::vspace::LogBarrierCostTerm<2>::MakeShared(vspace::vspace_error<2>(vel_var, V_MAX), weight));
            opt_problem.addCostTerm(steam::vspace::LogBarrierCostTerm<1>::MakeShared(
              vspace::vspace_error<1>(vspace::MatrixMultEvaluator<1, 2>::MakeShared(vel_var, A), b)
            , weight));
            opt_problem.addCostTerm(steam::vspace::LogBarrierCostTerm<2>::MakeShared(vspace::neg<2>(vspace::vspace_error<2>(vel_var, V_MIN)), weight));
        }



        // Solve the optimization problem with GaussNewton solver
        //using SolverType = steam::GaussNewtonSolver; // Old solver, does not have back stepping capability
        //using SolverType = steam::LineSearchGaussNewtonSolver;
        using SolverType = LevMarqGaussNewtonSolver;

        // Initialize solver parameters
        SolverType::Params params;
        params.verbose = true; // Makes the output display for debug when true
        params.max_iterations = 100;
        params.absolute_cost_change_threshold = 1e-2;

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
