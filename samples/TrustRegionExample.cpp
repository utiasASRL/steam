//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TrustRegionExample.cpp
/// \brief A sample usage of the STEAM Engine library for testing various trust-region solvers
///        on a divergent (for Gauss-Newton) error metric.
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <steam.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief A simple error metric designed to test trust region methods.
///        Implements a vector error in R^2, e[0] = x + 1, e[1] = -2*x^2 + x - 1.
///        Minimum error is at zero. Notably vanilla Gauss Newton is unable to converge to
///        the answer as a step near zero causes it to diverge.
//////////////////////////////////////////////////////////////////////////////////////////////
class DivergenceErrorEval : public steam::ErrorEvaluator
{
public:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Shared pointer typedefs for readability
  //////////////////////////////////////////////////////////////////////////////////////////////
  typedef boost::shared_ptr<DivergenceErrorEval> Ptr;
  typedef boost::shared_ptr<const DivergenceErrorEval> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Construct from a 1D state vector
  //////////////////////////////////////////////////////////////////////////////////////////////
  DivergenceErrorEval(const steam::VectorSpaceStateVar::ConstPtr& stateVec)
    : stateVec_(stateVec) {
    if (stateVec_->getPerturbDim() != 1) {
      throw std::invalid_argument("Dimension was improper size");
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const {
    return !stateVec_->isLocked();
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the error
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd evaluate() const {

    // Get value of state variable
    double x = stateVec_->getValue()[0];

    // Construct error and return
    Eigen::VectorXd v(2);
    v[0] = x + 1.0;
    v[1] = -2.0*x*x + x - 1.0;
    return v;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the error and Jacobians wrt state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::VectorXd evaluate(std::vector<steam::Jacobian>* jacs) const {

    // Get value of state variable
    double x = stateVec_->getValue()[0];

    // Check for null ptr and clear jacobians
    if (jacs == NULL) {
      throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
    }
    jacs->clear();

    // If state not locked, add Jacobian
    if(!stateVec_->isLocked()) {
      jacs->push_back(steam::Jacobian());
      (*jacs)[0].key = stateVec_->getKey();
      (*jacs)[0].jac = Eigen::MatrixXd(2,1);
      (*jacs)[0].jac(0,0) = 1.0;
      (*jacs)[0].jac(1,0) = -4.0*x + 1.0;
    }

    // Construct error and return
    Eigen::VectorXd v(2);
    v[0] = x + 1.0;
    v[1] = -2.0*x*x + x - 1.0;
    return v;
  }

private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Shared pointer to state variable
  //////////////////////////////////////////////////////////////////////////////////////////////
  steam::VectorSpaceStateVar::ConstPtr stateVec_;

};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Setup a fresh problem with unsolve variables
//////////////////////////////////////////////////////////////////////////////////////////////
void setupDivergenceProblem(steam::OptimizationProblem* problem) {

  // Create vector state variable
  Eigen::VectorXd initial(1);
  initial[0] = 10; // initial guess, solution is at zero...
  steam::VectorSpaceStateVar::Ptr stateVar(new steam::VectorSpaceStateVar(initial));

  // Setup shared noise and loss function
  steam::NoiseModel::Ptr sharedNoiseModel(new steam::NoiseModel(Eigen::MatrixXd::Identity(2,2)));
  steam::L2LossFunc::Ptr sharedLossFunc(new steam::L2LossFunc());

  // Setup cost term
  DivergenceErrorEval::Ptr errorfunc(new DivergenceErrorEval(stateVar));
  steam::CostTerm::Ptr costTerm(new steam::CostTerm(errorfunc, sharedNoiseModel, sharedLossFunc));

  // Init problem
  problem->addStateVariable(stateVar);
  problem->addCostTerm(costTerm);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Example of trying to solve the convergence problem with several solvers
//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

  // Solve using Vanilla Gauss-Newton Solver
  {
    steam::OptimizationProblem problem;
    setupDivergenceProblem(&problem);
    steam::VanillaGaussNewtonSolver::Params params; params.maxIterations = 100;
    steam::VanillaGaussNewtonSolver solver(&problem, params);
    solver.optimize();
    std::cout << "Vanilla Gauss Newton Terminates from: " << solver.getTerminationCause()
              << " after " << solver.getCurrIteration() << " iterations." << std::endl;
  }

  // Solve using Line Search Gauss-Newton Solver
  {
    steam::OptimizationProblem problem;
    setupDivergenceProblem(&problem);
    steam::LineSearchGaussNewtonSolver::Params params; params.maxIterations = 100;
    steam::LineSearchGaussNewtonSolver solver(&problem);
    solver.optimize();
    std::cout << "Line Search GN Terminates from: " << solver.getTerminationCause()
              << " after " << solver.getCurrIteration() << " iterations." << std::endl;
  }

  // Solve using Levenberg–Marquardt Solver
  {
    steam::OptimizationProblem problem;
    setupDivergenceProblem(&problem);
    steam::LevMarqGaussNewtonSolver::Params params; params.maxIterations = 100;
    steam::LevMarqGaussNewtonSolver solver(&problem);
    solver.optimize();
    std::cout << "Levenberg–Marquardt Terminates from: " << solver.getTerminationCause()
              << " after " << solver.getCurrIteration() << " iterations." << std::endl;
  }

  // Solve using Powell's Dogleg Solver
  {
    steam::OptimizationProblem problem;
    setupDivergenceProblem(&problem);
    steam::DoglegGaussNewtonSolver::Params params; params.maxIterations = 100;
    steam::DoglegGaussNewtonSolver solver(&problem);
    solver.optimize();
    std::cout << "Powell's Dogleg Terminates from: " << solver.getTerminationCause()
              << " after " << solver.getCurrIteration() << " iterations." << std::endl;
  }

  return 0;
}
