//////////////////////////////////////////////////////////////////////////////////////////////
/// \file EvalTreeHandle.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_EVAL_TREE_HANDLE_HPP
#define STEAM_EVAL_TREE_HANDLE_HPP

#include <steam/evaluator/blockauto/EvalTreeNode.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Simple container class for the block-automatic evaluation tree. The true purpose
///        of this class is to help handle the pool-memory release upon desctruction.
//////////////////////////////////////////////////////////////////////////////////////////////
template <typename TYPE>
class EvalTreeHandle
{
 public:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
#ifdef STEAM_USE_OBJECT_POOL
  EvalTreeHandle(EvalTreeNode<TYPE>* root) : root_(root) {}
#else
  EvalTreeHandle(typename EvalTreeNode<TYPE>::Ptr&& root) : root_(root) {}
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Destructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  ~EvalTreeHandle() {
#ifdef STEAM_USE_OBJECT_POOL
    EvalTreeNode<TYPE>::pool.returnObj(root_);
#endif
  }

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get root
  /////////////////////////////////////////////////////////////////////////////////////////////
#ifdef STEAM_USE_OBJECT_POOL
  EvalTreeNode<TYPE>* getRoot() const {
    return root_;
  }
#else
  typename EvalTreeNode<TYPE>::Ptr getRoot() const {
    return root_;
  }
#endif

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get current value
  /////////////////////////////////////////////////////////////////////////////////////////////
  const TYPE& getValue() const {
    return root_->getValue();
  }

 private:

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Instance of TYPE
  /////////////////////////////////////////////////////////////////////////////////////////////
#ifdef STEAM_USE_OBJECT_POOL
  EvalTreeNode<TYPE>* root_;
#else
  typename EvalTreeNode<TYPE>::Ptr root_;
#endif
};

} // steam

#endif // STEAM_EVAL_TREE_HANDLE_HPP
