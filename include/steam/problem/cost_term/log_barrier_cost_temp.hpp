#pragma once

#include <lgmath.hpp>

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/jacobians.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"


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
