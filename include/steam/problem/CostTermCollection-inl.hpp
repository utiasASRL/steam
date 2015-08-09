//////////////////////////////////////////////////////////////////////////////////////////////
/// \file CostTermCollection-inl.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/problem/CostTermCollection.hpp>

#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE>
CostTermCollection<MEAS_DIM,MAX_STATE_SIZE>::CostTermCollection() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a cost term
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE>
void CostTermCollection<MEAS_DIM,MAX_STATE_SIZE>::add(typename CostTerm<MEAS_DIM, MAX_STATE_SIZE>::ConstPtr costTerm) {
  costTerms_.push_back(costTerm);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compute the cost from the collection of cost terms
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE>
double CostTermCollection<MEAS_DIM,MAX_STATE_SIZE>::cost() const {

  // Calculate total cost in parallel
  double cost[NUMBER_OF_OPENMP_THREADS];
  #pragma omp parallel num_threads(NUMBER_OF_OPENMP_THREADS)
  {
    // Init costs
    int tid = omp_get_thread_num();
    cost[tid] = 0;

    #pragma omp for
    for(unsigned int i = 0; i < costTerms_.size(); i++) {
      cost[tid] += costTerms_.at(i)->evaluate();
    }
  }

  // Sum up costs and return total
  for(unsigned int i = 1; i < NUMBER_OF_OPENMP_THREADS; i++) {
    cost[0] += cost[i];
  }
  return cost[0];
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get size of the collection
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE>
size_t CostTermCollection<MEAS_DIM,MAX_STATE_SIZE>::size() const {
  return costTerms_.size();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get a reference to the cost terms
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE>
const std::vector<typename CostTerm<MEAS_DIM, MAX_STATE_SIZE>::ConstPtr>&
                          CostTermCollection<MEAS_DIM,MAX_STATE_SIZE>::getCostTerms() const {
  return costTerms_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Build the left-hand and right-hand sides of the Gauss-Newton system of equations
///        using the cost terms in this collection.
//////////////////////////////////////////////////////////////////////////////////////////////
template<int MEAS_DIM, int MAX_STATE_SIZE>
void CostTermCollection<MEAS_DIM,MAX_STATE_SIZE>::buildGaussNewtonTerms(
    const StateVector& stateVector,
    BlockSparseMatrix* approximateHessian,
    BlockVector* gradientVector) const {

  // Locally disable any internal eigen multithreading -- we do our own OpenMP
  Eigen::setNbThreads(1);

  // For each cost term
  #pragma omp parallel num_threads(NUMBER_OF_OPENMP_THREADS)
  {

    Eigen::Matrix<double,MAX_STATE_SIZE,MAX_STATE_SIZE> newHessianTerm;
    Eigen::Matrix<double,MAX_STATE_SIZE,1> newGradTerm;

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
        newGradTerm = (-1)*jacobians[i].jac.transpose()*error;

        // Update the right-hand side (thread critical)
        #pragma omp critical(b_update)
        {
          gradientVector->add(blkIdx1, newGradTerm);
        }

        // For each jacobian (in upper half)
        for (unsigned int j = i; j < jacobians.size(); j++) {

          // Get the key and state range affected
          unsigned int blkIdx2 = stateVector.getStateBlockIndex(jacobians[j].key);

          // Calculate terms needed to update the Gauss-Newton left-hand side
          unsigned int row;
          unsigned int col;
          if (blkIdx1 <= blkIdx2) {
            row = blkIdx1;
            col = blkIdx2;
            newHessianTerm = jacobians[i].jac.transpose()*jacobians[j].jac;
          } else {
            row = blkIdx2;
            col = blkIdx1;
            newHessianTerm = jacobians[j].jac.transpose()*jacobians[i].jac;
          }

          // Update the left-hand side (thread critical)
          #pragma omp critical(a_update)
          {
            approximateHessian->add(row, col, newHessianTerm);
          }
        }
      }
    }
  } // end parallel
}

} // steam
