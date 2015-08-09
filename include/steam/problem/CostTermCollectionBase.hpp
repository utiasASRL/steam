//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CostTermCollectionBase.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_COST_TERM_COLLECTION_BASE_HPP
#define STEAM_COST_TERM_COLLECTION_BASE_HPP

#include <boost/shared_ptr.hpp>

#include <steam/problem/CostTerm.hpp>

#include <steam/blockmat/BlockSparseMatrix.hpp>
#include <steam/blockmat/BlockVector.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Class that fully defines a nonlinear cost term (or 'factor').
///        Cost terms are composed of an error function, loss function and noise model.
//////////////////////////////////////////////////////////////////////////////////////////////
class CostTermCollectionBase
{
 public:

  /// Convenience typedefs
  typedef boost::shared_ptr<CostTermCollectionBase> Ptr;
  typedef boost::shared_ptr<const CostTermCollectionBase> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  CostTermCollectionBase() {}

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Build the left-hand and right-hand sides of the Gauss-Newton system of equations
  ///        using the cost terms in this collection.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void buildGaussNewtonTerms(const StateVector& stateVector,
                                     BlockSparseMatrix* approximateHessian,
                                     BlockVector* gradientVector) const = 0;
};

} // steam

#endif // STEAM_COST_TERM_COLLECTION_BASE_HPP
