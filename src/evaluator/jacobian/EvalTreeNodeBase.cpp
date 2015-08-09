//////////////////////////////////////////////////////////////////////////////////////////////
/// \file EvalTreeNodeBase.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/jacobian/EvalTreeNodeBase.hpp>

#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNodeBase::EvalTreeNodeBase() : index_(0) {
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
  for (unsigned i = 0; i < index_; i++) {
    children_[i]->release();
  }
  index_ = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add child evaluation node (order of addition is preserved)
//////////////////////////////////////////////////////////////////////////////////////////////
void EvalTreeNodeBase::addChild(EvalTreeNodeBase* newChild) {

  // Check for nullptr
  if (newChild != NULL) {

    if (index_ > 3) {
      throw std::runtime_error("Tried to add more than 4 children");
    }

    children_[index_] = newChild;
    index_++;

    //children_.push_back(newChild);
  } else {
    throw std::invalid_argument("Tried to add nullptr to EvalTreeNodeBase");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get number of child evaluation nodes
//////////////////////////////////////////////////////////////////////////////////////////////
size_t EvalTreeNodeBase::numChildren() const {
  return index_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get child node at index
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNodeBase* EvalTreeNodeBase::childAt(unsigned int index) const {

  if (index >= index_) {
    throw std::runtime_error("Tried to access null entry");
  }
  return children_[index];
}

} // steam
