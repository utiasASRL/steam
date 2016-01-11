//////////////////////////////////////////////////////////////////////////////////////////////
/// \file ParallelizedCostTermCollection-inl.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/ParallelizedCostTermCollection.hpp>

#include <iostream>
#include <steam/common/Timer.hpp>

#include <omp.h>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
template<int NUM_THREADS>
ParallelizedCostTermCollection<NUM_THREADS>::ParallelizedCostTermCollection() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a cost term
//////////////////////////////////////////////////////////////////////////////////////////////
template<int NUM_THREADS>
void ParallelizedCostTermCollection<NUM_THREADS>::add(const CostTermBase::ConstPtr& costTerm) {

  if (costTerm->isImplParallelized()) {
    throw std::runtime_error("Do not add pre-parallelized cost "
                             "terms to a cost term parallelizer.");
  }
  costTerms_.push_back(costTerm);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute the cost from the collection of cost terms
//////////////////////////////////////////////////////////////////////////////////////////////
template<int NUM_THREADS>
double ParallelizedCostTermCollection<NUM_THREADS>::cost() const {

  double cost = 0;
  #pragma omp parallel num_threads(NUM_THREADS)
  {
    #pragma omp for reduction(+:cost)
    for(unsigned int i = 0; i < costTerms_.size(); i++) {
      cost += costTerms_.at(i)->cost();
    }
  }
  return cost;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns the number of cost terms contained by this object
//////////////////////////////////////////////////////////////////////////////////////////////
template<int NUM_THREADS>
unsigned int ParallelizedCostTermCollection<NUM_THREADS>::numCostTerms() const {
  return costTerms_.size();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not the implementation already uses multi-threading
//////////////////////////////////////////////////////////////////////////////////////////////
template<int NUM_THREADS>
bool ParallelizedCostTermCollection<NUM_THREADS>::isImplParallelized() const {
  return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Build the left-hand and right-hand sides of the Gauss-Newton system of equations
///        using the cost terms in this collection.
//////////////////////////////////////////////////////////////////////////////////////////////
template<int NUM_THREADS>
void ParallelizedCostTermCollection<NUM_THREADS>::buildGaussNewtonTerms(
    const StateVector& stateVector,
    BlockSparseMatrix* approximateHessian,
    BlockVector* gradientVector) const {

  // Locally disable any internal eigen multithreading -- we do our own OpenMP
  Eigen::setNbThreads(1);

  // For each cost term
  #pragma omp parallel num_threads(NUM_THREADS)
  {
    #pragma omp for
    for (unsigned int c = 0 ; c < costTerms_.size(); c++) {

      costTerms_.at(c)->buildGaussNewtonTerms(stateVector, approximateHessian, gradientVector);

    } // end cost term loop
  } // end parallel
}

} // steam
