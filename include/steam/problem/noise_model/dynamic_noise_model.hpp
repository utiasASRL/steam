/**
 * \file DynamicNoiseModel.hpp
 * \author Sean Anderson, Alec Krawciw ASRL
 */
#pragma once

#include <iostream>

#include "steam/problem/noise_model/base_noise_model.hpp"
#include "steam/evaluable/evaluable.hpp"

namespace steam {


/**
 * \brief DynamicNoiseModel Noise model for uncertainties that change with the state variables
 * during the steam optimization problem.
 */
template <int DIM>
class DynamicNoiseModel : public BaseNoiseModel<DIM> {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<DynamicNoiseModel<DIM>>;
  using ConstPtr = std::shared_ptr<const DynamicNoiseModel<DIM>>;

  using MatrixT = Eigen::Matrix<double, DIM, DIM>;
  using VectorT = Eigen::Matrix<double, DIM, 1>;
  using MatrixTEvalPtr = typename Evaluable<MatrixT>::ConstPtr;

  static Ptr MakeShared(const MatrixTEvalPtr eval,
                        const NoiseType type = NoiseType::COVARIANCE);

  /**
   * \brief Constructor
   * \param[in] Evaluable<matrix> A noise matrix evaluable, determined by the type parameter.
   * \param[in] type The type of noise matrix set in the previous paramter.
   */
  DynamicNoiseModel(const MatrixTEvalPtr eval,
                   const NoiseType type = NoiseType::COVARIANCE);

  /** \brief Get a reference to the square root information matrix */
  MatrixT getSqrtInformation() const override;

  /**
   * \brief Get the norm of the whitened error vector,
   * sqrt(rawError^T * info * rawError)
   */
  double getWhitenedErrorNorm(const VectorT& rawError) const override;

  /**  \brief Get the whitened error vector, sqrtInformation * rawError */
  VectorT whitenError(const VectorT& rawError) const override;

 private:
  /** \brief Set by covariance matrix */
  MatrixT setByCovariance(const MatrixT& matrix) const;
  /** \brief Set by information matrix */
  MatrixT setByInformation(const MatrixT& matrix) const;
  /** \brief Set by square root of information matrix */
  MatrixT setBySqrtInformation(const MatrixT& matrix) const;

  /** \brief Assert that the matrix is positive definite */
  void assertPositiveDefiniteMatrix(const MatrixT& matrix) const;

  /**
   * \brief The square root information (found by performing an LLT
   * decomposition on the information matrix (inverse covariance matrix). This
   * triangular matrix is stored directly for faster error whitening.
   */
  const MatrixTEvalPtr eval_;
  NoiseType type_;
};



template <int DIM>
auto DynamicNoiseModel<DIM>::MakeShared(const MatrixTEvalPtr eval,
                                       const NoiseType type) -> Ptr {
  return std::make_shared<DynamicNoiseModel<DIM>>(eval, type);
}

template <int DIM>
DynamicNoiseModel<DIM>::DynamicNoiseModel(const MatrixTEvalPtr eval,
                                        const NoiseType type) : eval_(eval) {
                                          type_ = type;
                                          if (eval_ == nullptr)
                                            std::cout << "Evaluator initialization failed somehow";
                                        }

template <int DIM>
auto DynamicNoiseModel<DIM>::setByCovariance(const MatrixT& matrix) const -> MatrixT {
  // Information is the inverse of covariance
  return setByInformation(matrix.inverse());
}

template <int DIM>
auto DynamicNoiseModel<DIM>::setByInformation(const MatrixT& matrix) const -> MatrixT{
  // Check that the matrix is positive definite
  assertPositiveDefiniteMatrix(matrix);
  // Perform an LLT decomposition
  Eigen::LLT<MatrixT> lltOfInformation(matrix);
  // Store upper triangular matrix (the square root information matrix)
  return setBySqrtInformation(lltOfInformation.matrixL().transpose());
}

template <int DIM>
auto DynamicNoiseModel<DIM>::setBySqrtInformation(const MatrixT& matrix) const -> MatrixT{
  return matrix;  // todo: check this is upper triangular
}

template <int DIM>
auto DynamicNoiseModel<DIM>::getSqrtInformation() const -> MatrixT {
  const MatrixT matrix = eval_->value();
  switch (type_) {
    case NoiseType::INFORMATION:
      return setByInformation(matrix);
    case NoiseType::SQRT_INFORMATION:
      return setBySqrtInformation(matrix);
    case NoiseType::COVARIANCE:
    default:
      return setByCovariance(matrix);
  }
}

template <int DIM>
double DynamicNoiseModel<DIM>::getWhitenedErrorNorm(
    const VectorT& rawError) const {
  return (getSqrtInformation() * rawError).norm();
}

template <int DIM>
auto DynamicNoiseModel<DIM>::whitenError(const VectorT& rawError) const
    -> VectorT {
  return getSqrtInformation() * rawError;
}

template <int DIM>
void DynamicNoiseModel<DIM>::assertPositiveDefiniteMatrix(
    const MatrixT& matrix) const {
  // Initialize an eigen value solver
  Eigen::SelfAdjointEigenSolver<MatrixT> eigsolver(matrix,
                                                   Eigen::EigenvaluesOnly);

  // Check the minimum eigen value
  if (eigsolver.eigenvalues().minCoeff() <= 0) {
    std::stringstream ss;
    ss << "Covariance \n"
       << matrix << "\n must be positive definite. "
       << "Min. eigenvalue : " << eigsolver.eigenvalues().minCoeff() << std::endl;
    throw std::invalid_argument(ss.str());
  }
}

}  // namespace steam
