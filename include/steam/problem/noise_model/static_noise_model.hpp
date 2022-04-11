/**
 * \file StaticNoiseModel.hpp
 * \author Sean Anderson, ASRL
 */
#pragma once

#include "steam/problem/noise_model/base_noise_model.hpp"

namespace steam {

/** \brief Enumeration of ways to set the noise */
enum class NoiseType { COVARIANCE, INFORMATION, SQRT_INFORMATION };

/**
 * \brief StaticNoiseModel Noise model for uncertainties that do not change
 * during the steam optimization problem.
 */
template <int DIM>
class StaticNoiseModel : public BaseNoiseModel<DIM> {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<StaticNoiseModel<DIM>>;
  using ConstPtr = std::shared_ptr<const StaticNoiseModel<DIM>>;

  using MatrixT = typename BaseNoiseModel<DIM>::MatrixT;
  using VectorT = typename BaseNoiseModel<DIM>::VectorT;

  static Ptr MakeShared(const MatrixT& matrix,
                        const NoiseType& type = NoiseType::COVARIANCE);

  /**
   * \brief Constructor
   * \param[in] matrix A noise matrix, determined by the type parameter.
   * \param[in] type The type of noise matrix set in the previous paramter.
   */
  StaticNoiseModel(const MatrixT& matrix,
                   const NoiseType& type = NoiseType::COVARIANCE);

  /** \brief Set by covariance matrix */
  void setByCovariance(const MatrixT& matrix);
  /** \brief Set by information matrix */
  void setByInformation(const MatrixT& matrix);
  /** \brief Set by square root of information matrix */
  void setBySqrtInformation(const MatrixT& matrix);

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
  /** \brief Assert that the matrix is positive definite */
  void assertPositiveDefiniteMatrix(const MatrixT& matrix) const;

  /**
   * \brief The square root information (found by performing an LLT
   * decomposition on the information matrix (inverse covariance matrix). This
   * triangular matrix is stored directly for faster error whitening.
   */
  MatrixT sqrtInformation_;
};

template <int DIM>
auto StaticNoiseModel<DIM>::MakeShared(const MatrixT& matrix,
                                       const NoiseType& type) -> Ptr {
  return std::make_shared<StaticNoiseModel<DIM>>(matrix, type);
}

template <int DIM>
StaticNoiseModel<DIM>::StaticNoiseModel(const MatrixT& matrix,
                                        const NoiseType& type) {
  // Depending on the type of 'matrix', we set the internal storage
  switch (type) {
    case NoiseType::COVARIANCE:
      setByCovariance(matrix);
      break;
    case NoiseType::INFORMATION:
      setByInformation(matrix);
      break;
    case NoiseType::SQRT_INFORMATION:
      setBySqrtInformation(matrix);
      break;
  }
}

template <int DIM>
void StaticNoiseModel<DIM>::setByCovariance(const MatrixT& matrix) {
  // Information is the inverse of covariance
  setByInformation(matrix.inverse());
}

template <int DIM>
void StaticNoiseModel<DIM>::setByInformation(const MatrixT& matrix) {
  // Check that the matrix is positive definite
  assertPositiveDefiniteMatrix(matrix);
  // Perform an LLT decomposition
  Eigen::LLT<MatrixT> lltOfInformation(matrix);
  // Set internal storage matrix
  sqrtInformation_ = matrix;  // todo: check this is upper triangular
}

template <int DIM>
void StaticNoiseModel<DIM>::setBySqrtInformation(const MatrixT& matrix) {
  // Set internal storage matrix
  sqrtInformation_ = matrix;  // todo: check this is upper triangular
}

template <int DIM>
auto StaticNoiseModel<DIM>::getSqrtInformation() const -> MatrixT {
  return sqrtInformation_;
}

template <int DIM>
double StaticNoiseModel<DIM>::getWhitenedErrorNorm(
    const VectorT& rawError) const {
  return (sqrtInformation_ * rawError).norm();
}

template <int DIM>
auto StaticNoiseModel<DIM>::whitenError(const VectorT& rawError) const
    -> VectorT {
  return sqrtInformation_ * rawError;
}

template <int DIM>
void StaticNoiseModel<DIM>::assertPositiveDefiniteMatrix(
    const MatrixT& matrix) const {
  // Initialize an eigen value solver
  Eigen::SelfAdjointEigenSolver<MatrixT> eigsolver(matrix,
                                                   Eigen::EigenvaluesOnly);

  // Check the minimum eigen value
  if (eigsolver.eigenvalues().minCoeff() <= 0) {
    std::stringstream ss;
    ss << "Covariance \n"
       << matrix << "\n must be positive definite. "
       << "Min. eigenvalue : " << eigsolver.eigenvalues().minCoeff();
    throw std::invalid_argument(ss.str());
  }
}

}  // namespace steam
