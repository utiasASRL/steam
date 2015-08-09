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

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pool class
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename TYPE>
class Pool {

 public:

  // Default constructor
  Pool() : index_(0) {
    resources_.resize(10);
  }

  ~Pool() {
    //std::cout << "pool dying, made: " << numCreated_ << " dying with: " << resources_.size() << std::endl;
  }

  // Returns instance of Resource.
  TYPE* getObj()
  {
    /*if (resources_.empty())
    {
      //std::cout << "Creating new." << std::endl;
      numCreated_++;
      return new TYPE;
    }
    else
    {
      //std::cout << "Reusing existing." << std::endl;
      TYPE* resource = resources_.front();
      resources_.pop_front();
      return resource;
    }*/
    TYPE* result = &resources_[index_];
    index_ = (index_+1)%10;
    return result;
  }

  // Return resource back to the pool.
  void returnObj(TYPE* object)
  {
    object->reset();
    //resources_.push_back(object);
  }

 private:

  // todo - implement fixed size array... atleast if someone forgets to release its not a leak..
  std::vector<TYPE> resources_;
  unsigned int index_;

  //unsigned int numCreated_;

};

} // steam

#endif // STEAM_POOL_HPP
