#include "steam/problem/optimization_problem.hpp"

#include <iomanip>
#include <iostream>

namespace steam {

OptimizationProblem::OptimizationProblem(unsigned int num_threads)
    : num_threads_(num_threads) {}

void OptimizationProblem::addStateVariable(const StateVarBase::Ptr &state) {
  state_vars_.push_back(state);
}

void OptimizationProblem::addCostTerm(const BaseCostTerm::ConstPtr &costTerm) {
  cost_terms_.push_back(costTerm);
}

unsigned int OptimizationProblem::getNumberOfCostTerms() const {
  return cost_terms_.size();
}

double OptimizationProblem::cost() const {
  // Init
  double cost = 0;

  // Parallelize for the cost terms
#pragma omp parallel for reduction(+ : cost) num_threads(num_threads_)
  for (size_t i = 0; i < cost_terms_.size(); i++) {
    try {
      double cost_i = cost_terms_.at(i)->cost();
      if (std::isnan(cost_i)) {
        std::cout << "NaN cost term is ignored!" << std::endl;
      } else {
        cost += cost_i;
      }
    } catch (const std::exception &e) {
      std::cout << "STEAM exception in cost term:\n" << e.what() << std::endl;
    } catch (...) {
      std::cout << "STEAM exception in cost term: (unknown)" << std::endl;
    }
  }

  return cost;
}

StateVector &OptimizationProblem::getStateVector() {
  state_vector_ = StateVector();
  for (const auto &state_var : state_vars_) {
    if (!state_var->locked()) state_vector_.addStateVariable(state_var);
  }
  return state_vector_;
}

void OptimizationProblem::buildGaussNewtonTerms(
    Eigen::SparseMatrix<double> &approximate_hessian,
    Eigen::VectorXd &gradient_vector) {
  // Setup Matrices
  std::vector<unsigned int> sqSizes = state_vector_.getStateBlockSizes();
  BlockSparseMatrix A_(sqSizes, true);
  BlockVector b_(sqSizes);

  // Parallelize for the cost terms
#pragma omp parallel for num_threads(num_threads_)
  for (unsigned int c = 0; c < cost_terms_.size(); c++) {
    try {
      cost_terms_.at(c)->buildGaussNewtonTerms(state_vector_, &A_, &b_);
    } catch (const std::exception &e) {
      std::cout << "STEAM exception in parallel cost term:\n"
                << e.what() << std::endl;
    } catch (...) {
      std::cout << "STEAM exception in parallel cost term: (unknown)"
                << std::endl;
    }
  }  // end cost term loop

  // Convert to Eigen Type - with the block-sparsity pattern
  // ** Note we do not exploit sub-block-sparsity in case it changes at a later
  // iteration
  approximate_hessian = A_.toEigen(false);
  gradient_vector = b_.toEigen();
}

}  // namespace steam
