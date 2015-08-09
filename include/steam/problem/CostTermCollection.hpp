//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CostTermCollection.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_COST_TERM_COLLECTION_HPP
#define STEAM_COST_TERM_COLLECTION_HPP

#include <boost/shared_ptr.hpp>

#include <steam/problem/CostTermCollectionBase.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Class that fully defines a nonlinear cost term (or 'factor').
///        Cost terms are composed of an error function, loss function and noise model.
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE>
class CostTermCollection : public CostTermCollectionBase
{
 public:

  /// Convenience typedefs
  typedef boost::shared_ptr<CostTermCollection<MEAS_DIM, MAX_STATE_SIZE> > Ptr;
  typedef boost::shared_ptr<const CostTermCollection<MEAS_DIM, MAX_STATE_SIZE> > ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  CostTermCollection();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Add a cost term
  //////////////////////////////////////////////////////////////////////////////////////////////
  void add(typename CostTerm<MEAS_DIM, MAX_STATE_SIZE>::ConstPtr costTerm);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Compute the cost from the collection of cost terms
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual double cost() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get size of the collection
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual size_t size() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get a reference to the cost terms
  //////////////////////////////////////////////////////////////////////////////////////////////
  const std::vector<typename CostTerm<MEAS_DIM, MAX_STATE_SIZE>::ConstPtr>& getCostTerms() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Build the left-hand and right-hand sides of the Gauss-Newton system of equations
  ///        using the cost terms in this collection.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void buildGaussNewtonTerms(const StateVector& stateVector,
                                     BlockSparseMatrix* approximateHessian,
                                     BlockVector* gradientVector) const;

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Collection of nonlinear cost-term factors
  //////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<typename CostTerm<MEAS_DIM, MAX_STATE_SIZE>::ConstPtr> costTerms_;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Typedef for the general dynamic cost term
//////////////////////////////////////////////////////////////////////////////////////////////
typedef CostTermCollection<Eigen::Dynamic, Eigen::Dynamic> CostTermCollectionX;

} // steam

#include <steam/problem/CostTermCollection-inl.hpp>

#endif // STEAM_COST_TERM_COLLECTION_HPP
