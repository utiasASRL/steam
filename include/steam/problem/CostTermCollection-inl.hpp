//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CostTermCollection-inl.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/CostTermCollection.hpp>

#include <iostream>
#include <steam/common/Timer.hpp>

#include <omp.h>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE, int NUM_THREADS>
CostTermCollection<MEAS_DIM,MAX_STATE_SIZE,NUM_THREADS>::CostTermCollection() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a cost term
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE, int NUM_THREADS>
void CostTermCollection<MEAS_DIM,MAX_STATE_SIZE,NUM_THREADS>::add(
    typename CostTerm<MEAS_DIM, MAX_STATE_SIZE>::ConstPtr costTerm) {
  costTerms_.push_back(costTerm);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute the cost from the collection of cost terms
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE, int NUM_THREADS>
double CostTermCollection<MEAS_DIM,MAX_STATE_SIZE,NUM_THREADS>::cost() const {

  double cost = 0;
  #pragma omp parallel num_threads(NUM_THREADS)
  {
    #pragma omp for reduction(+:cost)
    for(unsigned int i = 0; i < costTerms_.size(); i++) {
      cost += costTerms_.at(i)->evaluate();
    }
  }
  return cost;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get size of the collection
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE, int NUM_THREADS>
size_t CostTermCollection<MEAS_DIM,MAX_STATE_SIZE,NUM_THREADS>::size() const {
  return costTerms_.size();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get a reference to the cost terms
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE, int NUM_THREADS>
const std::vector<typename CostTerm<MEAS_DIM, MAX_STATE_SIZE>::ConstPtr>&
    CostTermCollection<MEAS_DIM,MAX_STATE_SIZE,NUM_THREADS>::getCostTerms() const {
  return costTerms_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Build the left-hand and right-hand sides of the Gauss-Newton system of equations
///        using the cost terms in this collection.
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE, int NUM_THREADS>
void CostTermCollection<MEAS_DIM,MAX_STATE_SIZE,NUM_THREADS>::buildGaussNewtonTerms(
    const StateVector& stateVector,
    BlockSparseMatrix* approximateHessian,
    BlockVector* gradientVector) const {

  // Locally disable any internal eigen multithreading -- we do our own OpenMP
  Eigen::setNbThreads(1);

  // Get square block indices (we know the hessian is block-symmetric)
  const std::vector<unsigned int>& blkSizes =
      approximateHessian->getIndexing().rowIndexing().blkSizes();

  // For each cost term
  #pragma omp parallel num_threads(NUM_THREADS)
  {

    // Init dynamic matrices
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,0,MAX_STATE_SIZE,MAX_STATE_SIZE> newHessianTerm;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,0,MAX_STATE_SIZE,1> newGradTerm;

    #pragma omp for
    for (unsigned int c = 0 ; c < costTerms_.size(); c++) {

      // Compute the weighted and whitened errors and jacobians
      // err = sqrt(w)*sqrt(R^-1)*rawError
      // jac = sqrt(w)*sqrt(R^-1)*rawJacobian
      std::vector<Jacobian<MEAS_DIM,MAX_STATE_SIZE> > jacobians;
      Eigen::Matrix<double,MEAS_DIM,1> error = costTerms_.at(c)->evalWeightedAndWhitened(&jacobians);

      // For each jacobian
      for (unsigned int i = 0; i < jacobians.size(); i++) {

        // Get the key and state range affected
        unsigned int blkIdx1 = stateVector.getStateBlockIndex(jacobians[i].key);

        // Calculate terms needed to update the right-hand-side
        unsigned int size1 = blkSizes.at(blkIdx1);
        newGradTerm = (-1)*jacobians[i].jac.leftCols(size1).transpose()*error;

        // Update the right-hand side (thread critical)
        #pragma omp critical(b_update)
        {
          gradientVector->mapAt(blkIdx1) += newGradTerm;
        }

        // For each jacobian (in upper half)
        for (unsigned int j = i; j < jacobians.size(); j++) {

          // Get the key and state range affected
          unsigned int blkIdx2 = stateVector.getStateBlockIndex(jacobians[j].key);

          // Calculate terms needed to update the Gauss-Newton left-hand side
          unsigned int size2 = blkSizes.at(blkIdx2);
          unsigned int row;
          unsigned int col;
          if (blkIdx1 <= blkIdx2) {
            row = blkIdx1;
            col = blkIdx2;
            newHessianTerm = jacobians[i].jac.leftCols(size1).transpose()*jacobians[j].jac.leftCols(size2);
          } else {
            row = blkIdx2;
            col = blkIdx1;
            newHessianTerm = jacobians[j].jac.leftCols(size2).transpose()*jacobians[i].jac.leftCols(size1);
          }

          // Update the left-hand side (thread critical)
          BlockSparseMatrix::BlockRowEntry& entry = approximateHessian->rowEntryAt(row, col, true);
          omp_set_lock(&entry.lock);
          entry.data += newHessianTerm;
          omp_unset_lock(&entry.lock);

        } // end row loop
      } // end column loop
    } // end cost term loop
  } // end parallel
}

} // steam
