/**
 * \file BaseNoiseModel.hpp
 * \author Sean Anderson, Yuchen Wu, ASRL
 */
#pragma once

#include <Eigen/Dense>

namespace steam {

/** \brief Enumeration of ways to set the noise */
enum class NoiseType { COVARIANCE, INFORMATION, SQRT_INFORMATION };

/** \brief BaseNoiseModel Base class for the steam noise models */
template <int DIM>
class BaseNoiseModel {
 public:
  /** \brief Convenience typedefs */
  using Ptr = std::shared_ptr<BaseNoiseModel<DIM>>;
  using ConstPtr = std::shared_ptr<const BaseNoiseModel<DIM>>;

  using MatrixT = Eigen::Matrix<double, DIM, DIM>;
  using VectorT = Eigen::Matrix<double, DIM, 1>;

  virtual ~BaseNoiseModel() = default;

  /** \brief Get a reference to the square root information matrix */
  virtual MatrixT getSqrtInformation() const = 0;

  /**
   * \brief Get the norm of the whitened error vector,
   * sqrt(rawError^T * info * rawError)
   */
  virtual double getWhitenedErrorNorm(const VectorT& rawError) const = 0;

  /** \brief Get the whitened error vector, sqrtInformation*rawError */
  virtual VectorT whitenError(const VectorT& rawError) const = 0;
};

}  // namespace steam

