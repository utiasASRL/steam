//////////////////////////////////////////////////////////////////////////////////////////////
/// \file Pool.hpp
/// \brief Implements a basic singleton object pool. The implementation is fairly naive,
///        but should be fast given its assumptions. The OmpPool is also thread safe for
///        OpenMP threads, assuming the number of OpenMP threads was set at compile time.
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_POOL_HPP
#define STEAM_POOL_HPP

#include <iostream>
#include <vector>
#include <stdexcept>

#include <omp.h>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pool class
///
/// While the object pool class could be implemented with a linked list, we choose to use
/// a simple array of max size for efficiency. It makes the logic a bit simpler, but more
/// importantly, if someone forgets to return an object, we do not have memory leaks.
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE, int MAX_SIZE = 50>
class Pool {
 public:

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  ////////////////////////////////////////////////////////////////////////////////////////////
  Pool() {
    index_ = 0;
    resources_ = new TYPE[MAX_SIZE];
    for (unsigned int i = 0; i < MAX_SIZE; i++) {
      available_[i] = true;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Destructor
  ////////////////////////////////////////////////////////////////////////////////////////////
  ~Pool() {
    if (resources_) {
      delete [] resources_;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get an object
  ////////////////////////////////////////////////////////////////////////////////////////////
  TYPE* getObj() {

    // Check if current is availble
    if (!available_[index_]) {

      // Increment index (wraps at end)
      index_ = ((MAX_SIZE-1) == index_) ? 0 : (index_+1);

      // Loop over entire array once
      unsigned int i = 0;
      for (; i < MAX_SIZE; i++) {
        if (available_[index_]) {
          // Found an available object, break out and return it
          break;
        } else {
          // Increment index (wraps at end)
          index_ = ((MAX_SIZE-1) == index_) ? 0 : (index_+1);
        }
      }

      // Check that we found an available resource, otherwise we need to throw a runtime error
      if (i == MAX_SIZE) {
        throw std::runtime_error("Pool ran out of entries... make sure they are being released.");
      }
    }

    // Mark as not available and give away resource
    available_[index_] = false;
    TYPE* result = &resources_[index_];
    index_ = ((MAX_SIZE-1) == index_) ? 0 : (index_+1);
    return result;
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Return an object to the pool
  ////////////////////////////////////////////////////////////////////////////////////////////
  void returnObj(TYPE* object) {

    // Reset the objects data
    object->reset();

    // Calculate the index from the pointer
    std::ptrdiff_t index = object - &resources_[0];

    // Set its available to true
    available_[index] = true;
  }

 private:

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Array of objects
  ////////////////////////////////////////////////////////////////////////////////////////////
  TYPE* resources_;

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Array of availability
  ////////////////////////////////////////////////////////////////////////////////////////////
  bool available_[MAX_SIZE];

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Current index of next most likely available resource
  ////////////////////////////////////////////////////////////////////////////////////////////
  unsigned int index_;
};


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief OpenMP enabled pool class. This is implemented fairly naively by taking advantage
///        of the number of OpenMP threads at compile time. By having a seperate pool for
///        each thread, we are fully safe from sychronization issues.
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE, int MAX_SIZE = 50>
class OmpPool {
 public:

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  ////////////////////////////////////////////////////////////////////////////////////////////
  OmpPool() {

    // Initialize pointers
    for (int i = 0; i < MAX_NUM_THREADS; i++) {
      pools_[i] = NULL;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Destructor
  ////////////////////////////////////////////////////////////////////////////////////////////
  ~OmpPool() {

    // Deallocate pools
    for (int i = 0; i < MAX_NUM_THREADS; i++) {
      if (pools_[i]) {
        delete pools_[i];
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get an object
  ////////////////////////////////////////////////////////////////////////////////////////////
  TYPE* getObj() {

    // Unique OpenMP thread identifier
    int tid = omp_get_thread_num();

    // Get thread identifier is in range
    if (tid < 0 || tid >= MAX_NUM_THREADS) {
      throw std::runtime_error("Thread ID is higher than maximum number of threads allowed in pool.");
    }

    // Check if we have allocated a pool for this thread
    if (pools_[tid] == NULL) {
      pools_[tid] = new Pool<TYPE,MAX_SIZE>();
    }

    // Return result from pool
    return pools_[tid]->getObj();
  }

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Return an object to the pool
  ////////////////////////////////////////////////////////////////////////////////////////////
  void returnObj(TYPE* object) {

    // Unique OpenMP thread identifier
    int tid = omp_get_thread_num();

    // Get thread identifier is in range
    if (tid < 0 || tid >= MAX_NUM_THREADS) {
      throw std::runtime_error("Thread ID is higher than maximum number of threads allowed in pool.");
    }

    // Check if we have allocated a pool for this thread
    if (pools_[tid] == NULL) {
      throw std::runtime_error("Resource returned to an OpenMP Pool that does not exist.");
    }

    // Return object to pool
    pools_[tid]->returnObj(object);
  }

 private:

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Maximum number of threads this pool can support.
  ////////////////////////////////////////////////////////////////////////////////////////////
  static const int MAX_NUM_THREADS = 64;

  ////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Array of pools (one for each "possible" thread)
  ////////////////////////////////////////////////////////////////////////////////////////////
  Pool<TYPE,MAX_SIZE>* pools_[MAX_NUM_THREADS];
};

} // steam

#endif // STEAM_POOL_HPP
