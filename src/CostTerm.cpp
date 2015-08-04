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
//Eigen::VectorXd CostTerm::evalWeightedAndWhitened(std::vector<Jacobian>* outJacobians) const {

//  // Get raw error and Jacobians
//  std::pair<Eigen::VectorXd, JacobianTreeNode::ConstPtr> rawError = errorFunction_->evaluateJacobians();

//  // Get whitened error vector
//  Eigen::VectorXd whiteError = noiseModel_->whitenError(rawError.first);

//  // Get weight from loss function
//  double sqrt_w = sqrt(lossFunc_->weight(whiteError.norm()));

//  // Check and initialize jacobian array
//  if (outJacobians == NULL) {
//    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
//  }
//  outJacobians->clear();

//  // Get Jacobians
//  rawError.second->append(sqrt_w * noiseModel_->getSqrtInformation(), outJacobians);

//  // Weight the error and return
//  return sqrt_w * whiteError;
//}

Eigen::VectorXd CostTerm::evalWeightedAndWhitened(std::vector<Jacobian>* outJacobians) const {

  // Get raw error and Jacobians
  std::vector<Jacobian> rawJacs;
  Eigen::VectorXd rawError = errorFunction_->evaluate(&rawJacs);

  // Check and initialize jacobian array
  if (outJacobians == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  outJacobians->clear();
  outJacobians->resize(rawJacs.size());

  // Get whitened error vector
  Eigen::VectorXd whiteError = noiseModel_->whitenError(rawError);

  // Get weight from loss function
  double sqrt_w = sqrt(lossFunc_->weight(whiteError.norm()));

  // Whiten and weight the Jacobians
  for (unsigned int j = 0; j < rawJacs.size(); j++) {

    // Check for dimension mismatch
    if (noiseModel_->getSqrtInformation().cols() != rawJacs[j].jac.rows()) {
      throw std::runtime_error("Dimension mismatch");
    }

    // Whiten jacobian and weight by loss function
    Jacobian& jacref = outJacobians->at(j);
    jacref.key = rawJacs[j].key;
    jacref.jac = sqrt_w * noiseModel_->getSqrtInformation() * rawJacs[j].jac;
  }

  // Weight the error and return
  return sqrt_w * whiteError;
}

} // steam
