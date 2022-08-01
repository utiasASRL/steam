#include "steam/problem/sliding_window_filter.hpp"

#include <iomanip>
#include <iostream>

namespace steam {

auto SlidingWindowFilter::MakeShared(unsigned int num_threads)
    -> SlidingWindowFilter::Ptr {
  return std::make_shared<SlidingWindowFilter>(num_threads);
}

SlidingWindowFilter::SlidingWindowFilter(unsigned int num_threads)
    : num_threads_(num_threads) {}

void SlidingWindowFilter::addStateVariable(const StateVarBase::Ptr &variable) {
  addStateVariable(std::vector<StateVarBase::Ptr>{variable});
}

void SlidingWindowFilter::addStateVariable(
    const std::vector<StateVarBase::Ptr> &variables) {
  for (const auto &variable : variables) {
    const auto res = variables_.try_emplace(variable->key(), variable, false);
    if (!res.second) throw std::runtime_error("duplicated variable key");
    variable_queue_.emplace_back(variable->key());
    related_var_keys_.try_emplace(variable->key(), KeySet{variable->key()});
  }
}

void SlidingWindowFilter::marginalizeVariable(
    const StateVarBase::Ptr &variable) {
  marginalizeVariable(std::vector<StateVarBase::Ptr>{variable});
}

void SlidingWindowFilter::marginalizeVariable(
    const std::vector<StateVarBase::Ptr> &variables) {
  if (variables.empty()) return;

  ///
  for (const auto &variable : variables) {
    variables_.at(variable->key()).marginalize = true;
  }

  /// remove fixed variables from the queue

  StateVector fixed_state_vector;
  StateVector state_vector;

  //
  std::vector<StateKey> to_remove;
  bool fixed = true;
  for (const auto &key : variable_queue_) {
    const auto &var = variables_.at(key);
    const auto &related_keys = related_var_keys_.at(key);
    if (std::all_of(related_keys.begin(), related_keys.end(),
                    [this](const StateKey &key) {
                      return variables_.at(key).marginalize;
                    })) {
      if (!fixed)
        throw std::runtime_error("fixed variables must be at the first");
      fixed_state_vector.addStateVariable(var.variable);
      to_remove.emplace_back(key);
    } else {
      fixed = false;
    }
    state_vector.addStateVariable(var.variable);
  }

  //
  std::vector<BaseCostTerm::ConstPtr> active_cost_terms;
  active_cost_terms.reserve(cost_terms_.size());
  //
  const auto state_sizes = state_vector.getStateBlockSizes();
  BlockSparseMatrix A_(state_sizes, true);
  BlockVector b_(state_sizes);
#pragma omp parallel for num_threads(num_threads_)
  for (unsigned int c = 0; c < cost_terms_.size(); c++) {
    KeySet keys;
    cost_terms_.at(c)->getRelatedVarKeys(keys);
    if (std::all_of(keys.begin(), keys.end(), [this](const StateKey &key) {
          return variables_.at(key).marginalize;
        })) {
      cost_terms_.at(c)->buildGaussNewtonTerms(state_vector, &A_, &b_);
    } else {
#pragma omp critical(active_cost_terms_update)
      { active_cost_terms.emplace_back(cost_terms_.at(c)); }
    }
  }
  //
  cost_terms_ = active_cost_terms;

  /// \todo use sparse matrix
  Eigen::MatrixXd Aupper(A_.toEigen(false));
  Eigen::MatrixXd A(Aupper.selfadjointView<Eigen::Upper>());
  Eigen::VectorXd b(b_.toEigen());

  // add the cached terms (always top-left block)
  if (fixed_A_.size() > 0) {
    A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
    b.head(fixed_b_.size()) += fixed_b_;
  }

  // marginalize the fixed variables
  const auto fixed_state_size = fixed_state_vector.getStateSize();
  if (fixed_state_size > 0) {
    // clang-format off
    Eigen::MatrixXd A00(A.topLeftCorner(fixed_state_size, fixed_state_size));
    Eigen::MatrixXd A10(A.bottomLeftCorner(A.rows() - fixed_state_size, fixed_state_size));
    Eigen::MatrixXd A11(A.bottomRightCorner(A.rows() - fixed_state_size, A.cols() - fixed_state_size));
    Eigen::VectorXd b0(b.head(fixed_state_size));
    Eigen::VectorXd b1(b.tail(b.size() - fixed_state_size));
    fixed_A_ = A11 - A10 * A00.inverse() * A10.transpose();
    fixed_b_ = b1 - A10 * A00.inverse() * b0;
    // clang-format on
  } else {
    fixed_A_ = A;
    fixed_b_ = b;
  }

  /// remove the fixed variables
  for (const auto &key : to_remove) {
    const auto related_keys = related_var_keys_.at(key);
    for (const auto &related_key : related_keys) {
      related_var_keys_.at(related_key).erase(key);
    }
    related_var_keys_.erase(key);
    variables_.erase(key);
    if (variable_queue_.empty() || variable_queue_.front() != key)
      throw std::runtime_error("variable queue is not consistent");
    variable_queue_.pop_front();
  }
}

void SlidingWindowFilter::addCostTerm(const BaseCostTerm::ConstPtr &cost_term) {
  cost_terms_.emplace_back(cost_term);

  KeySet related_keys;
  cost_term->getRelatedVarKeys(related_keys);
  for (const auto &key : related_keys) {
    related_var_keys_.at(key).insert(related_keys.begin(), related_keys.end());
  }
}

double SlidingWindowFilter::cost() const {
  // Init
  double cost = 0;

  // Parallelize for the cost terms
#pragma omp parallel for reduction(+ : cost) num_threads(num_threads_)
  for (size_t i = 0; i < cost_terms_.size(); i++) {
    double cost_i = cost_terms_.at(i)->cost();
    if (std::isnan(cost_i)) {
      std::cout << "NaN cost term is ignored!" << std::endl;
    } else {
      cost += cost_i;
    }
  }

  return cost;
}

unsigned int SlidingWindowFilter::getNumberOfCostTerms() const {
  return cost_terms_.size();
}

unsigned int SlidingWindowFilter::getNumberOfVariables() const {
  return variable_queue_.size();
}

StateVector::Ptr SlidingWindowFilter::getStateVector() const {
  *marginalize_state_vector_ = StateVector();
  *active_state_vector_ = StateVector();
  *state_vector_ = StateVector();

  bool marginalize = true;
  for (const auto &key : variable_queue_) {
    const auto &var = variables_.at(key);
    if (var.marginalize) {
      if (!marginalize)
        throw std::runtime_error("marginalized variables must be at the first");
      marginalize_state_vector_->addStateVariable(var.variable);
    } else {
      marginalize = false;
      active_state_vector_->addStateVariable(var.variable);
    }
    state_vector_->addStateVariable(var.variable);
  }

  return active_state_vector_;
}

void SlidingWindowFilter::buildGaussNewtonTerms(
    Eigen::SparseMatrix<double> &approximate_hessian,
    Eigen::VectorXd &gradient_vector) const {
  //
  std::vector<unsigned int> sqSizes = state_vector_->getStateBlockSizes();
  BlockSparseMatrix A_(sqSizes, true);
  BlockVector b_(sqSizes);
#pragma omp parallel for num_threads(num_threads_)
  for (unsigned int c = 0; c < cost_terms_.size(); c++) {
    cost_terms_.at(c)->buildGaussNewtonTerms(*state_vector_, &A_, &b_);
  }

  // Convert to Eigen Types
  Eigen::MatrixXd Aupper(A_.toEigen(false));
  Eigen::MatrixXd A(Aupper.selfadjointView<Eigen::Upper>());
  Eigen::VectorXd b(b_.toEigen());

  if (fixed_A_.size() > 0) {
    A.topLeftCorner(fixed_A_.rows(), fixed_A_.cols()) += fixed_A_;
    b.head(fixed_b_.size()) += fixed_b_;
  }

  // marginalize the fixed variables
  const auto marginalize_state_size = marginalize_state_vector_->getStateSize();
  if (marginalize_state_size > 0) {
    // clang-format off
    Eigen::MatrixXd A00(A.topLeftCorner(marginalize_state_size, marginalize_state_size));
    Eigen::MatrixXd A10(A.bottomLeftCorner(A.rows() - marginalize_state_size, marginalize_state_size));
    Eigen::MatrixXd A11(A.bottomRightCorner(A.rows() - marginalize_state_size, A.cols() - marginalize_state_size));
    Eigen::VectorXd b0(b.head(marginalize_state_size));
    Eigen::VectorXd b1(b.tail(b.size() - marginalize_state_size));
    approximate_hessian = Eigen::MatrixXd(A11 - A10 * A00.inverse() * A10.transpose()).sparseView();
    gradient_vector = b1 - A10 * A00.inverse() * b0;
    // clang-format on
  } else {
    approximate_hessian = A.sparseView();
    gradient_vector = b;
  }
}

}  // namespace steam
