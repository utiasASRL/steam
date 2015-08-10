//////////////////////////////////////////////////////////////////////////////////////////////
/// \file LossFunctions.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/LossFunctions.hpp>

namespace steam {


//////////////////////////////////////////////////////////////////////////////////////////////
/// L2 Loss Function
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Cost function (basic evaluation of the loss function)
//////////////////////////////////////////////////////////////////////////////////////////////
double L2LossFunc::cost(double whitened_error_norm) const {
  return 0.5*whitened_error_norm*whitened_error_norm;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Weight for iteratively reweighted least-squares (influence function div. by error)
//////////////////////////////////////////////////////////////////////////////////////////////
double L2LossFunc::weight(double whitened_error_norm) const {
  return 1.0;
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// Huber Loss Function
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Cost function (basic evaluation of the loss function)
//////////////////////////////////////////////////////////////////////////////////////////////
double HuberLossFunc::cost(double whitened_error_norm) const {
  double e2 = whitened_error_norm*whitened_error_norm;
  double abse = fabs(whitened_error_norm); // should already be positive anyway ...
  if (abse <= k_) {
    return 0.5*e2;
  } else {
    return k_*(abse - 0.5*k_);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Weight for iteratively reweighted least-squares (influence function div. by error)
//////////////////////////////////////////////////////////////////////////////////////////////
double HuberLossFunc::weight(double whitened_error_norm) const {
  double abse = fabs(whitened_error_norm); // should already be positive anyway ...
  if (abse <= k_) {
    return 1.0;
  } else {
    return k_/abse;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// DCS Loss Function
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Cost function (basic evaluation of the loss function)
//////////////////////////////////////////////////////////////////////////////////////////////
double DcsLossFunc::cost(double whitened_error_norm) const {
  double e2 = whitened_error_norm*whitened_error_norm;
  if (e2 <= k2_) {
    return 0.5*e2;
  } else {
    return 2.0*k2_*e2/(k2_+e2) - 0.5*k2_;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Weight for iteratively reweighted least-squares (influence function div. by error)
//////////////////////////////////////////////////////////////////////////////////////////////
double DcsLossFunc::weight(double whitened_error_norm) const {
  double e2 = whitened_error_norm*whitened_error_norm;
  if (e2 <= k2_) {
    return 1.0;
  } else {
    double k2e2 = k2_ + e2;
    return 4.0*k2_*k2_/(k2e2*k2e2);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Geman-McClure Loss Function
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Cost function (basic evaluation of the loss function)
//////////////////////////////////////////////////////////////////////////////////////////////
double GemanMcClureLossFunc::cost(double whitened_error_norm) const {
  double e2 = whitened_error_norm*whitened_error_norm;
  return 0.5 * e2 / (k2_ + e2);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Weight for iteratively reweighted least-squares (influence function div. by error)
//////////////////////////////////////////////////////////////////////////////////////////////
double GemanMcClureLossFunc::weight(double whitened_error_norm) const {
  double e2 = whitened_error_norm*whitened_error_norm;
  double k2e2 = k2_ + e2;
  return k2_*k2_/(k2e2*k2e2);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// Cauchy Loss Function
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Cost function (basic evaluation of the loss function)
//////////////////////////////////////////////////////////////////////////////////////////////
double CauchyLossFunc::cost(double whitened_error_norm) const {
  double e_div_k = fabs(whitened_error_norm)/k_;
  return 0.5 * k_ * k_ * std::log(1.0 + e_div_k*e_div_k);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Weight for iteratively reweighted least-squares (influence function div. by error)
//////////////////////////////////////////////////////////////////////////////////////////////
double CauchyLossFunc::weight(double whitened_error_norm) const {
  double e_div_k = fabs(whitened_error_norm)/k_;
  return 1.0 / (1.0 + e_div_k*e_div_k);
}

} // steam