//////////////////////////////////////////////////////////////////////////////////////////////
/// \file Pool.hpp
/// \brief Implements a basic singleton object pool. The implementation is fairly naive,
///        but should be fast given its assumptions. It is also thread safe assuming the
///        number of threads is know at compile Pool.
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_POOL_HPP
#define STEAM_POOL_HPP

#include <iostream>
#include <vector>
#include <stdexcept>

namespace steam {

//template <typename T, int N>
//class Array
//{
//  public:
//    T& operator[](int index)
//    {
//      // add check for array index out of bounds i.e. access within 0 to N-1
//      return data[index];
//    }
//    Array() {
//        data = new T[size = N];
//    }
//    ~Array() {
//        if (data)
//            delete [] data;
//    }
//  private:
//    int size;
//    T *data;
//};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pool class
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE, int MAX_SIZE = 100>
class Pool {
 public:

  Pool() {
    resources_ = new TYPE[MAX_SIZE];
    for (unsigned int i = 0; i < MAX_SIZE; i++) {
      available_[i] = true;
    }
  }
  ~Pool() {
    if (resources_)
      delete [] resources_;
  }

  // Returns instance of Resource.
  TYPE* getObj()
  {
    if (!available_[index_]) {
      index_ = ((MAX_SIZE-1) == index_) ? 0 : (index_+1);
      unsigned int i = 0;
      for (; i < MAX_SIZE; i++) {
        if (available_[index_]) {
          break;
        } else {
          index_ = ((MAX_SIZE-1) == index_) ? 0 : (index_+1);
        }
      }
      if (i == MAX_SIZE) {
        throw std::runtime_error("Pool ran out of entries... make sure they are being released.");
      }
    }

    //int tid = omp_get_thread_num();
    //std::cout << tid;

    // Mark as not available and give away resource
    available_[index_] = false;
    TYPE* result = &resources_[index_];
    index_ = ((MAX_SIZE-1) == index_) ? 0 : (index_+1);
    return result;
  }

  // Return resource back to the pool.
  void returnObj(TYPE* object)
  {
    object->reset();
    std::ptrdiff_t index = object - &resources_[0];
    available_[index] = true;
  }

 private:

  // data
  TYPE* resources_;

  // availability
  bool available_[MAX_SIZE];

  // current index of next most likely available resource
  unsigned int index_;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pool class for use with OpenMP multithreads
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE, int MAX_SIZE = 100>
class OmpPool {
 public:

  OmpPool() {}
  ~OmpPool() {}

  // Returns instance of Resource.
  TYPE* getObj()
  {
    int tid = omp_get_thread_num();
    //std::cout << tid;
    return pools_[tid].getObj();
  }

  // Return resource back to the pool.
  void returnObj(TYPE* object)
  {
    int tid = omp_get_thread_num();
    pools_[tid].returnObj(object);
  }

 private:

  Pool<TYPE,MAX_SIZE> pools_[NUMBER_OF_OPENMP_THREADS];
};

} // steam

#endif // STEAM_POOL_HPP
