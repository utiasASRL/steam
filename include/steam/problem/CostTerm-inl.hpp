//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CostTerm-inl.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/CostTerm.hpp>

#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
template <int MEAS_DIM, int MAX_STATE_SIZE>
CostTerm<MEAS_DIM,MAX_STATE_SIZE>::CostTerm(
    const typename ErrorEvaluator<MEAS_DIM,MAX_STATE_SIZE>::ConstPtr& errorFunction,
    const typename NoiseModel<MEAS_DIM>::ConstPtr& noiseModel,
    const LossFunction::ConstPtr& lossFunc) :
  errorFunction_(errorFunction), noiseModel_(noiseModel), lossFunc_(lossFunc) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the cost of this term. Error is first whitened by the noise model
///        and then passed through the loss function, as in
///          cost = loss(sqrt(e^T * cov^{-1} * e))
//////////////////////////////////////////////////////////////////////////////////////////////
template <int MEAS_DIM, int MAX_STATE_SIZE>
double CostTerm<MEAS_DIM,MAX_STATE_SIZE>::evaluate() const
{
  return lossFunc_->cost(noiseModel_->getWhitenedErrorNorm(errorFunction_->evaluate()));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the iteratively reweighted error vector and Jacobians. The error and
///        Jacobians are first whitened by the noise model and then weighted by the loss
///        function, as in:
///              error = sqrt(weight)*sqrt(cov^-1)*rawError
///           jacobian = sqrt(weight)*sqrt(cov^-1)*rawJacobian
//////////////////////////////////////////////////////////////////////////////////////////////
template <int MEAS_DIM, int MAX_STATE_SIZE>
Eigen::Matrix<double,MEAS_DIM,1> CostTerm<MEAS_DIM,MAX_STATE_SIZE>::evalWeightedAndWhitened(
    std::vector<Jacobian<MEAS_DIM,MAX_STATE_SIZE> >* outJacobians) const {

  // Check and initialize jacobian array
  if (outJacobians == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  outJacobians->clear();

  // Get raw error and Jacobians
  Eigen::Matrix<double,MEAS_DIM,1> rawError =
      errorFunction_->evaluate(noiseModel_->getSqrtInformation(), outJacobians);

  // Get whitened error vector
  Eigen::Matrix<double,MEAS_DIM,1> whiteError = noiseModel_->whitenError(rawError);

  // Get weight from loss function
  double sqrt_w = sqrt(lossFunc_->weight(whiteError.norm()));

  // Weight the white jacobians
  for (unsigned int i = 0; i < outJacobians->size(); i++) {
    outJacobians->at(i).jac *= sqrt_w;
  }

  // Weight the error and return
  return sqrt_w * whiteError;
}

} // steam
