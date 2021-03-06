//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CauchyLossFunc.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/lossfunc/CauchyLossFunc.hpp>

namespace steam {

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
