//////////////////////////////////////////////////////////////////////////////////////////////
/// \file NoiseModel.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_NOISE_MODEL_HPP
#define STEAM_NOISE_MODEL_HPP

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>

namespace steam {

/// Noise Model
template <int MEAS_DIM>
class NoiseModel
{
 public:

  /// Enumeration of ways to set the noise
  enum MatrixType { COVARIANCE, INFORMATION, SQRT_INFORMATION };

  /// Convenience typedefs
  typedef boost::shared_ptr<NoiseModel<MEAS_DIM> > Ptr;
  typedef boost::shared_ptr<const NoiseModel<MEAS_DIM> > ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  NoiseModel();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief General constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  NoiseModel(const Eigen::Matrix<double,MEAS_DIM,MEAS_DIM>& matrix, MatrixType type = COVARIANCE);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Set by covariance matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  void setByCovariance(const Eigen::Matrix<double,MEAS_DIM,MEAS_DIM>& matrix);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Set by information matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  void setByInformation(const Eigen::Matrix<double,MEAS_DIM,MEAS_DIM>& matrix);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Set by square root of information matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  void setBySqrtInformation(const Eigen::Matrix<double,MEAS_DIM,MEAS_DIM>& matrix);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get a reference to the square root information matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  const Eigen::Matrix<double,MEAS_DIM,MEAS_DIM>& getSqrtInformation() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get the norm of the whitened error vector, sqrt(rawError^T * info * rawError)
  //////////////////////////////////////////////////////////////////////////////////////////////
  double getWhitenedErrorNorm(const Eigen::Matrix<double,MEAS_DIM,1>& rawError) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get the whitened error vector, sqrtInformation*rawError
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double,MEAS_DIM,1> whitenError(const Eigen::Matrix<double,MEAS_DIM,1>& rawError) const;

private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Assert that the matrix is positive definite
  //////////////////////////////////////////////////////////////////////////////////////////////
  void assertPositiveDefiniteMatrix(const Eigen::Matrix<double,MEAS_DIM,MEAS_DIM>& matrix);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief The square root information (found by performing an LLT decomposition on the
  ///        information matrix (inverse covariance matrix). This triangular matrix is
  ///        stored directly for faster error whitening.
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double,MEAS_DIM,MEAS_DIM> sqrtInformation_;

};

typedef NoiseModel<Eigen::Dynamic> NoiseModelX;

} // steam

#include <steam/NoiseModel-inl.hpp>

#endif // STEAM_NOISE_MODEL_HPP
