//////////////////////////////////////////////////////////////////////////////////////////////
/// \file EvalTreeNodeBase.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/blockauto/EvalTreeNodeBase.hpp>

#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNodeBase::EvalTreeNodeBase() : numChildren_(0) {
  for (unsigned i = 0; i < MAX_NUM_CHILDREN_; i++) {
    children_[i] = NULL;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Destructor
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNodeBase::~EvalTreeNodeBase() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Release children and reset internal indexing to zero.
//////////////////////////////////////////////////////////////////////////////////////////////
void EvalTreeNodeBase::reset() {
#ifdef STEAM_USE_OBJECT_POOL
  for (unsigned i = 0; i < numChildren_; i++) {
    children_[i]->release();
  }
#endif
  for (unsigned i = 0; i < MAX_NUM_CHILDREN_; i++) {
    children_[i] = NULL;
  }
  numChildren_ = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add child evaluation node (order of addition is preserved)
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef STEAM_USE_OBJECT_POOL
void EvalTreeNodeBase::addChild(EvalTreeNodeBase* newChild) {
#else
void EvalTreeNodeBase::addChild(EvalTreeNodeBase::Ptr newChild) {
#endif

  // Check for nullptr
  if (newChild != NULL) {

    if (numChildren_ >= MAX_NUM_CHILDREN_) {
      throw std::runtime_error("Tried to add more than the maximum number of children");
    }

    children_[numChildren_] = newChild;

    numChildren_++;

  } else {
    throw std::invalid_argument("Tried to add nullptr to EvalTreeNodeBase");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get number of child evaluation nodes
//////////////////////////////////////////////////////////////////////////////////////////////
size_t EvalTreeNodeBase::numChildren() const {
  return numChildren_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get child node at index
//////////////////////////////////////////////////////////////////////////////////////////////
#ifdef STEAM_USE_OBJECT_POOL
EvalTreeNodeBase* EvalTreeNodeBase::childAt(unsigned int index) const {
#else
EvalTreeNodeBase::Ptr EvalTreeNodeBase::childAt(unsigned int index) const{
#endif

  if (index >= numChildren_) {
    throw std::runtime_error("Tried to access null entry");
  }
  return children_[index];
}

} // steam
