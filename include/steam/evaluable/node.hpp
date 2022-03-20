#pragma once

#include <memory>
#include <vector>

namespace steam {

class NodeBase {
 public:
  using Ptr = std::shared_ptr<NodeBase>;
  using ConstPtr = std::shared_ptr<const NodeBase>;

  virtual ~NodeBase() = default;

  /** \brief Adds a child node (order of addition is preserved) */
  void addChild(const Ptr& child) { children_.emplace_back(child); }

  /** \brief Returns child at index */
  Ptr at(const size_t& index) const { return children_[index]; }

 private:
  std::vector<Ptr> children_;
};

template <class T>
class Node : public NodeBase {
 public:
  using Ptr = std::shared_ptr<Node<T>>;
  using ConstPtr = std::shared_ptr<const Node<T>>;

  static Ptr MakeShared(const T& value) {
    return std::make_shared<Node<T>>(value);
  }

  Node(const T& value) : value_(value) {}

  const T& value() const { return value_; }

 private:
  T value_;
};

}  // namespace steam