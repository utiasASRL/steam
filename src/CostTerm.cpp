//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CostTerm.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/CostTerm.hpp>
#include <glog/logging.h>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
CostTerm::CostTerm(const ErrorEvaluator::ConstPtr& errorFunction, const NoiseModel::ConstPtr& noiseModel, const LossFunction::ConstPtr& lossFunc) :
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
Eigen::VectorXd CostTerm::evalWeightedAndWhitened(std::vector<Jacobian>* outJacobians) const {

  // Get raw error and Jacobians
  Eigen::VectorXd whiteError = noiseModel_->whitenError(errorFunction_->evaluate(outJacobians));

  // Get weight from loss function
  double sqrt_w = sqrt(lossFunc_->weight(whiteError.norm()));

  // Whiten and weight the Jacobians
  for (unsigned int i = 0; i < outJacobians->size(); i++) {
    CHECK(noiseModel_->getSqrtInformation().cols() == (*outJacobians)[i].jac.rows()); // TODO, remove this check, and instead check dimensions of all 'parties' on construction of object...
    (*outJacobians)[i].jac = sqrt_w*noiseModel_->getSqrtInformation()*(*outJacobians)[i].jac;
  }

  // Weight the error and return
  whiteError *= sqrt_w;
  return whiteError;
}

} // steam
