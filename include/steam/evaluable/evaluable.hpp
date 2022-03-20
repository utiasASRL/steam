#pragma once

#include <Eigen/Core>

#include "steam/evaluable/jacobians.hpp"
#include "steam/evaluable/node.hpp"

namespace steam {

template <class T>
class Evaluable {
 public:
  using Ptr = std::shared_ptr<Evaluable<T>>;
  using ConstPtr = std::shared_ptr<const Evaluable<T>>;

  virtual ~Evaluable() = default;

  T evaluate() const { return this->forward()->value(); }
  T evaluate(const Eigen::MatrixXd& lhs, Jacobians& jacs) const {
    const auto end_node = this->forward();
    backward(lhs, end_node, jacs);
    return end_node->value();
  }

  virtual bool active() const = 0;
  virtual typename Node<T>::Ptr forward() const = 0;
  virtual void backward(const Eigen::MatrixXd& lhs,
                        const typename Node<T>::Ptr& node,
                        Jacobians& jacs) const = 0;
};

}  // namespace steam