#pragma once

#include <Eigen/Dense>

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/state_key.hpp"

namespace steam {

class StateVarBase {
 public:
  using Ptr = std::shared_ptr<StateVarBase>;
  using ConstPtr = std::shared_ptr<const StateVarBase>;

  StateVarBase(const unsigned int& perturb_dim, const std::string& name = "")
      : perturb_dim_(perturb_dim), name_(name) {}
  virtual ~StateVarBase() = default;

  std::string name() const { return name_; }

  /** \brief Updates this state from a perturbation */
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
  const std::string name_;
  const StateKey key_ = NewStateKey();
  bool locked_ = false;
};

template <class T>
class StateVar : public StateVarBase, public Evaluable<T> {
 public:
  using Ptr = std::shared_ptr<StateVar<T>>;
  using ConstPtr = std::shared_ptr<const StateVar<T>>;

  StateVar(const T& value, const unsigned int& perturb_dim,
           const std::string& name = "")
      : StateVarBase(perturb_dim, name), value_(value) {}

  void setFromCopy(const StateVarBase::ConstPtr& other) override {
    if (key() != other->key())
      throw std::runtime_error("StateVar::setFromCopy: keys do not match");
    value_ = std::static_pointer_cast<const StateVar<T>>(other)->value_;
  }

  bool active() const override { return !locked(); }
  using KeySet = typename Evaluable<T>::KeySet;
  void getRelatedVarKeys(KeySet& keys) const override {
    if (!locked()) keys.insert(key());
  }
  T value() const override { return value_; }
  typename Node<T>::Ptr forward() const override {
    return Node<T>::MakeShared(value_);
  }
  void backward(const Eigen::MatrixXd& lhs, const typename Node<T>::Ptr& node,
                Jacobians& jacs) const override {
    if (active()) jacs.add(key(), lhs);
  }

 protected:
  T value_;
};

}  // namespace steam