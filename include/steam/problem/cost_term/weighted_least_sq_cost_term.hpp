#pragma once

#include <algorithm>

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/jacobians.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/loss_func/base_loss_func.hpp"
#include "steam/problem/noise_model/base_noise_model.hpp"

namespace steam {

template <int DIM>
class WeightedLeastSqCostTerm : public BaseCostTerm {
 public:
  using Ptr = std::shared_ptr<WeightedLeastSqCostTerm<DIM>>;
  using ConstPtr = std::shared_ptr<const WeightedLeastSqCostTerm<DIM>>;

  using ErrorType = Eigen::Matrix<double, DIM, 1>;  // DIM is measurement dim

  static Ptr MakeShared(
      const typename Evaluable<ErrorType>::ConstPtr &error_function,
      const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
      const BaseLossFunc::ConstPtr &loss_function,
      const std::string &name = "");
  WeightedLeastSqCostTerm(
      const typename Evaluable<ErrorType>::ConstPtr &error_function,
      const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
      const BaseLossFunc::ConstPtr &loss_function,
      const std::string &name = "");

  /**
   * \brief Evaluates the cost of this term. Error is first whitened by the
   * noise model and then passed through the loss function, as in:
   *     cost = loss(sqrt(e^T * cov^{-1} * e))
   */
  double cost() const override;

  std::string name() const override {
    return name_;
  }

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
  BaseLossFunc::ConstPtr loss_function_;

};

template <int DIM>
auto WeightedLeastSqCostTerm<DIM>::MakeShared(
    const typename Evaluable<ErrorType>::ConstPtr &error_function,
    const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
    const BaseLossFunc::ConstPtr &loss_function,
    const std::string &name) -> Ptr {
  return std::make_shared<WeightedLeastSqCostTerm<DIM>>(
      error_function, noise_model, loss_function, name);
}

template <int DIM>
WeightedLeastSqCostTerm<DIM>::WeightedLeastSqCostTerm(
    const typename Evaluable<ErrorType>::ConstPtr &error_function,
    const typename BaseNoiseModel<DIM>::ConstPtr &noise_model,
    const BaseLossFunc::ConstPtr &loss_function,
    const std::string &name)
    : error_function_(error_function),
      noise_model_(noise_model),
      loss_function_(loss_function) {
        name_ = name;
      }

template <int DIM>
double WeightedLeastSqCostTerm<DIM>::cost() const {
  return loss_function_->cost(
      noise_model_->getWhitenedErrorNorm(error_function_->evaluate()));
}

template <int DIM>
void WeightedLeastSqCostTerm<DIM>::getRelatedVarKeys(KeySet &keys) const {
  error_function_->getRelatedVarKeys(keys);
}

template <int DIM>
void WeightedLeastSqCostTerm<DIM>::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
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
    Eigen::MatrixXd newGradTerm = (-1) * jac1.transpose() * error;

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
      unsigned int row, col;
      const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
        if (blkIdx1 <= blkIdx2) {
          row = blkIdx1;
          col = blkIdx2;
          return jac1.transpose() * jac2;
        } else {
          row = blkIdx2;
          col = blkIdx1;
          return jac2.transpose() * jac1;
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
