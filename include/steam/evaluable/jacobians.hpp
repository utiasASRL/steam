#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "steam/evaluable/state_key.hpp"

namespace steam {

class Jacobians {
 public:
  using KeyJacMap = std::unordered_map<StateKey, Eigen::MatrixXd, StateKeyHash>;

  /** \brief Adds a child node (order of addition is preserved) */
  void add(const StateKey &key, const Eigen::MatrixXd &jac) {
    auto iter_success = jacs_.try_emplace(key, jac);
    if (!iter_success.second) iter_success.first->second += jac;
  }

  void clear() { jacs_.clear(); }

  KeyJacMap &get() { return jacs_; }

 private:
  KeyJacMap jacs_;
};

}  // namespace steam