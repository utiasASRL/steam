#pragma once

#include <algorithm>

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/jacobians.hpp"
#include "steam/problem/CostTermBase.hpp"
#include "steam/problem/LossFunctions.hpp"
#include "steam/problem/NoiseModel.hpp"

namespace steam {

template <int DIM>
class WeightedLeastSqCostTerm : public CostTermBase {
 public:
  using Ptr = std::shared_ptr<WeightedLeastSqCostTerm<DIM>>;
  using ConstPtr = std::shared_ptr<const WeightedLeastSqCostTerm<DIM>>;

  using ErrorType = Eigen::Matrix<double, DIM, 1>;  // DIM is measurement dim

  WeightedLeastSqCostTerm(
      const typename Evaluable<ErrorType>::ConstPtr &error_function,
      const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
      const LossFunctionBase::ConstPtr &loss_function);

  /**
   * \brief Evaluates the cost of this term. Error is first whitened by the
   * noise model and then passed through the loss function, as in:
   *     cost = loss(sqrt(e^T * cov^{-1} * e))
   */
  double cost() const override;

  /**
   * \brief Add the contribution of this cost term to the left-hand (Hessian)
   * and right-hand (gradient vector) sides of the Gauss-Newton system of
   * equations.
   */
  void buildGaussNewtonTerms(const StateVector &state_vec,
                             BlockSparseMatrix *approximate_hessian,
                             BlockVector *gradient_vector) const override;

 private:
  /**
   * \brief Evaluate the iteratively reweighted error vector and Jacobians. The
   * error and Jacobians are first whitened by the noise model and then
   * weighted by the loss function, as in:
   *              error = sqrt(weight)*sqrt(cov^-1)*rawError
   *           jacobian = sqrt(weight)*sqrt(cov^-1)*rawJacobian
   */
  ErrorType evalWeightedAndWhitened(Jacobians &jacobian_contaner) const;

  /** \brief Error evaluator */
  typename Evaluable<ErrorType>::ConstPtr error_function_;
  /** \brief Noise model */
  typename BaseNoiseModel<DIM>::ConstPtr noise_model_;
  /** \brief Loss function */
  LossFunctionBase::ConstPtr loss_function_;
};

template <int DIM>
WeightedLeastSqCostTerm<DIM>::WeightedLeastSqCostTerm(
    const typename Evaluable<ErrorType>::ConstPtr &error_function,
    const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
    const LossFunctionBase::ConstPtr &loss_function)
    : error_function_(error_function),
      noise_model_(noise_model),
      loss_function_(loss_function) {}

template <int DIM>
double WeightedLeastSqCostTerm<DIM>::cost() const {
  return loss_function_->cost(
      noise_model_->getWhitenedErrorNorm(error_function_->evaluate()));
}

template <int DIM>
void WeightedLeastSqCostTerm<DIM>::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  // Get square block indices (we know the hessian is block-symmetric)
  const std::vector<unsigned int> &blkSizes =
      approximate_hessian->getIndexing().rowIndexing().blkSizes();

  // Init dynamic matrices
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> newHessianTerm;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> newGradTerm;

  // Compute the weighted and whitened errors and jacobians
  // err = sqrt(w)*sqrt(R^-1)*rawError
  // jac = sqrt(w)*sqrt(R^-1)*rawJacobian
  Jacobians jacobian_contaner;
  ErrorType error = this->evalWeightedAndWhitened(jacobian_contaner);
  const auto &jacobians = jacobian_contaner.get();

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

    // Calculate terms needed to update the right-hand-side
    unsigned int size1 = blkSizes.at(blkIdx1);
    newGradTerm = (-1) * jac1.leftCols(size1).transpose() * error;

// Update the right-hand side (thread critical)
#pragma omp critical(b_update)
    { gradient_vector->mapAt(blkIdx1) += newGradTerm; }

    // For each jacobian (in upper half)
    for (size_t j = i; j < keys.size(); j++) {
      const auto &key2 = keys.at(j);
      const auto &jac2 = jacobians.at(key2);

      // Get the key and state range affected
      unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

      // Calculate terms needed to update the Gauss-Newton left-hand side
      unsigned int size2 = blkSizes.at(blkIdx2);
      unsigned int row;
      unsigned int col;
      if (blkIdx1 <= blkIdx2) {
        row = blkIdx1;
        col = blkIdx2;
        newHessianTerm =
            jac1.leftCols(size1).transpose() * jac2.leftCols(size2);
      } else {
        row = blkIdx2;
        col = blkIdx1;
        newHessianTerm =
            jac2.leftCols(size2).transpose() * jac1.leftCols(size1);
      }

      // Update the left-hand side (thread critical)
      BlockSparseMatrix::BlockRowEntry &entry =
          approximate_hessian->rowEntryAt(row, col, true);
      omp_set_lock(&entry.lock);
      entry.data += newHessianTerm;
      omp_unset_lock(&entry.lock);

    }  // end row loop
  }    // end column loop
}

template <int DIM>
auto WeightedLeastSqCostTerm<DIM>::evalWeightedAndWhitened(
    Jacobians &jacobian_contaner) const -> ErrorType {
  // initializes jacobian array
  jacobian_contaner.clear();

  // Get raw error and Jacobians
  ErrorType raw_error = error_function_->evaluate(
      noise_model_->getSqrtInformation(), jacobian_contaner);

  // Get whitened error vector
  ErrorType white_error = noise_model_->whitenError(raw_error);

  // Get weight from loss function
  double sqrt_w = sqrt(loss_function_->weight(white_error.norm()));

  // Weight the white jacobians
  auto &jacobians = jacobian_contaner.get();
  for (auto &entry : jacobians) entry.second *= sqrt_w;

  // Weight the error and return
  return sqrt_w * white_error;
}

}  // namespace steam
