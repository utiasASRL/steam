//////////////////////////////////////////////////////////////////////////////////////////////
/// \file L2LossFunc.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/lossfunc/L2LossFunc.hpp>

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

} // steam
