#pragma once

#include <Eigen/Dense>

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/state_key.hpp"

namespace steam {

class StateVarBase {
 public:
  using Ptr = std::shared_ptr<StateVarBase>;
  using ConstPtr = std::shared_ptr<const StateVarBase>;

  StateVarBase(const unsigned int& perturb_dim) : perturb_dim_(perturb_dim) {}

  virtual ~StateVarBase() = default;

  /** \brief Updates a state from a perturbation */
  virtual bool update(const Eigen::VectorXd& perturbation) = 0;
  /** \brief Returns a clone of the state */
  virtual Ptr clone() const = 0;
  /** \brief Sets the state value from anotehr instance of the state */
  virtual void setFromCopy(const ConstPtr& other) = 0;

  const StateKey& key() const { return key_; }
  const unsigned int& perturb_dim() const { return perturb_dim_; }
  const bool& locked() const { return locked_; }
  bool& locked() { return locked_; }

 private:
  const unsigned int perturb_dim_;
  const StateKey key_ = NewStateKey();
  bool locked_ = false;
};

template <class T>
class StateVar : public StateVarBase, public Evaluable<T> {
 public:
  using Ptr = std::shared_ptr<StateVar<T>>;
  using ConstPtr = std::shared_ptr<const StateVar<T>>;

  StateVar(const T& value, const unsigned int& perturb_dim)
      : StateVarBase(perturb_dim), value_(value) {}

  void setFromCopy(const StateVarBase::ConstPtr& other) override {
    if (key() != other->key())
      throw std::runtime_error("StateVar::setFromCopy: keys do not match");
    value_ = std::static_pointer_cast<const StateVar<T>>(other)->value_;
  }

  const T& getValue() const { return value_; }
  void setValue(const T& value) { value_ = value; }

  bool active() const override { return !locked(); }

 protected:
  T value_;
};

}  // namespace steam