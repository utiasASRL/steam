////////////////////////////////////////////////////////////////////////////////
/// \file TrustRegionExample.cpp
/// \brief A sample usage of the STEAM Engine library for testing various
/// trust-region solvers on a divergent (for Gauss-Newton) error metric.
///
/// \author Sean Anderson, ASRL
////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <steam.hpp>

/**
 * \brief A simple error metric designed to test trust region methods.
 * Implements a vector error in R^2, e[0] = x + 1, e[1] = -2*x^2 + x - 1.
 * Minimum error is at zero. Notably vanilla Gauss Newton is unable to converge
 * to the answer as a step near zero causes it to diverge.
 */
class DivergenceErrorEval
    : public steam::Evaluable<Eigen::Matrix<double, 2, 1>> {
 public:
  /** \brief Shared pointer typedefs for readability */
  using Ptr = std::shared_ptr<DivergenceErrorEval>;
  using ConstPtr = std::shared_ptr<const DivergenceErrorEval>;

  /** \brief Input/Output typedefs */
  using InType = Eigen::Matrix<double, 1, 1>;
  using OutType = Eigen::Matrix<double, 2, 1>;

  static Ptr MakeShared(const steam::Evaluable<InType>::ConstPtr& input) {
    return std::make_shared<DivergenceErrorEval>(input);
  }

  /** \brief Construct from a 1D state vector */
  DivergenceErrorEval(const steam::Evaluable<InType>::ConstPtr& input)
      : input_(input) {}

  /** \brief Returns whether or not an evaluator contains unlocked states */
  bool active() const override { return input_->active(); }

  OutType value() const override {
    double x = input_->value().value();
    OutType v;
    v(0, 0) = x + 1.0;
    v(1, 0) = -2.0 * x * x + x - 1.0;
    return v;
  }

  /** \brief Evaluate the error, i.e. forward pass */
  steam::Node<OutType>::Ptr forward() const override {
    // Get node from input evaluable
    const auto child = input_->forward();

    // Construct error
    double x = child->value().value();

    OutType v;
    v(0, 0) = x + 1.0;
    v(1, 0) = -2.0 * x * x + x - 1.0;

    // Construct node and add child
    const auto node = steam::Node<OutType>::MakeShared(v);
    node->addChild(child);

    return node;
  }

  /** \brief Evaluates the jacobian, i.e. backward pass */
  void backward(const Eigen::MatrixXd& lhs,
                const steam::Node<OutType>::Ptr& node,
                steam::Jacobians& jacs) const {
    // If state is active, add Jacobian
    if (input_->active()) {
      // Get child node
      const auto child =
          std::static_pointer_cast<steam::Node<InType>>(node->at(0));

      // Get Input Value
      double x = child->value().value();

      // Fill out matrix
      Eigen::MatrixXd jacobian(2, 1);
      jacobian(0, 0) = 1.0;
      jacobian(1, 0) = -4.0 * x + 1.0;
      Eigen::MatrixXd new_lhs = lhs * jacobian;

      input_->backward(new_lhs, child, jacs);
    }
  }

 private:
  /** \brief Shared pointer to input evaluable */
  const steam::Evaluable<InType>::ConstPtr input_;
};

/** \brief Setup a fresh problem with unsolve variables */
steam::OptimizationProblem setupDivergenceProblem() {
  // Create vector state variable
  Eigen::Matrix<double, 1, 1> initial(1);
  initial(0, 0) = 10;  // initial guess, solution is at zero...
  const auto state_var = steam::vspace::VSpaceStateVar<1>::MakeShared(initial);

  // Setup shared noise and loss function
  Eigen::Matrix2d cov = Eigen::Matrix2d::Identity();
  const auto noise_model = steam::StaticNoiseModel<2>::MakeShared(cov);
  const auto loss_function = steam::L2LossFunc::MakeShared();

  // Error function
  const auto error_function = DivergenceErrorEval::MakeShared(state_var);

  // Setup cost term
  const auto cost_term = steam::WeightedLeastSqCostTerm<2>::MakeShared(
      error_function, noise_model, loss_function);

  // Init problem
  steam::OptimizationProblem problem;
  problem.addStateVariable(state_var);
  problem.addCostTerm(cost_term);

  return problem;
}

/** \brief Example of trying to solve the problem with several solvers */
int main(int argc, char** argv) {
  // Solve using Vanilla Gauss-Newton Solver
  {
    steam::OptimizationProblem problem = setupDivergenceProblem();
    steam::VanillaGaussNewtonSolver::Params params;
    params.maxIterations = 100;
    steam::VanillaGaussNewtonSolver solver(&problem, params);
    solver.optimize();
    std::cout << "Vanilla Gauss Newton Terminates from: "
              << solver.getTerminationCause() << " after "
              << solver.getCurrIteration() << " iterations." << std::endl;
  }

  // Solve using Line Search Gauss-Newton Solver
  {
    steam::OptimizationProblem problem = setupDivergenceProblem();
    steam::LineSearchGaussNewtonSolver::Params params;
    params.maxIterations = 100;
    steam::LineSearchGaussNewtonSolver solver(&problem, params);
    solver.optimize();
    std::cout << "Line Search GN Terminates from: "
              << solver.getTerminationCause() << " after "
              << solver.getCurrIteration() << " iterations." << std::endl;
  }

  // Solve using Levenberg–Marquardt Solver
  {
    steam::OptimizationProblem problem = setupDivergenceProblem();
    steam::LevMarqGaussNewtonSolver::Params params;
    params.maxIterations = 100;
    steam::LevMarqGaussNewtonSolver solver(&problem, params);
    solver.optimize();
    std::cout << "Levenberg–Marquardt Terminates from: "
              << solver.getTerminationCause() << " after "
              << solver.getCurrIteration() << " iterations." << std::endl;
  }

  // Solve using Powell's Dogleg Solver
  {
    steam::OptimizationProblem problem = setupDivergenceProblem();
    steam::DoglegGaussNewtonSolver::Params params;
    params.maxIterations = 100;
    steam::DoglegGaussNewtonSolver solver(&problem, params);
    solver.optimize();
    std::cout << "Powell's Dogleg Terminates from: "
              << solver.getTerminationCause() << " after "
              << solver.getCurrIteration() << " iterations." << std::endl;
  }

  return 0;
}
