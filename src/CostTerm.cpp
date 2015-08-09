//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CostTerm.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/CostTerm.hpp>

#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
CostTerm::CostTerm(const ErrorEvaluatorX::ConstPtr& errorFunction, const NoiseModelX::ConstPtr& noiseModel, const LossFunction::ConstPtr& lossFunc) :
  errorFunction_(errorFunction), noiseModel_(noiseModel), lossFunc_(lossFunc) {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the cost of this term. Error is first whitened by the noise model
///        and then passed through the loss function, as in
///          cost = loss(sqrt(e^T * cov^{-1} * e))
//////////////////////////////////////////////////////////////////////////////////////////////
double CostTerm::evaluate() const
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
Eigen::VectorXd CostTerm::evalWeightedAndWhitened(std::vector<Jacobian<> >* outJacobians) const {

  // Check and initialize jacobian array
  if (outJacobians == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  outJacobians->clear();

  // Get raw error and Jacobians
  Eigen::VectorXd rawError = errorFunction_->evaluate(noiseModel_->getSqrtInformation(), outJacobians);

  // Get whitened error vector
  Eigen::VectorXd whiteError = noiseModel_->whitenError(rawError);

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
