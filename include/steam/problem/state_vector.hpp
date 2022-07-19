#pragma once

#include <map>
#include <vector>

#include "steam/evaluable/state_key.hpp"
#include "steam/evaluable/state_var.hpp"

namespace steam {

class StateVector {
 public:
  /** \brief Performs a deep copy of the state vector */
  StateVector clone() const;

  /**
   * \brief Copy the values of 'other' into 'this'
   * \note states must already align,typically this means that one is already a
   * deep copy of the other
   */
  void copyValues(const StateVector &other);

  /** \brief Add state variable */
  void addStateVariable(const StateVarBase::Ptr &statevar);

  /** \brief Check if a state variable exists in the vector */
  bool hasStateVariable(const StateKey &key) const;

  /** \brief Get a state variable using a key */
  StateVarBase::ConstPtr getStateVariable(const StateKey &key) const;

  /** \brief Get number of state variables */
  unsigned int getNumberOfStates() const;

  /** \brief Get the block index of a state */
  int getStateBlockIndex(const StateKey &key) const;

  /** \brief Get an ordered list of the sizes of the 'block' state variables */
  std::vector<unsigned int> getStateBlockSizes() const;

  /** \brief Total size of the state vector */
  unsigned int getStateSize() const;

  /** \brief Update the state vector */
  void update(const Eigen::VectorXd &perturbation);

 private:
  /** \brief Container of state-related and indexing variables */
  struct StateContainer {
    /// State
    StateVarBase::Ptr state;

    /// Block index in active state (set to -1 if not an active variable)
    int local_block_index;
  };

  /** \brief Main container for state variables */
  std::unordered_map<StateKey, StateContainer, StateKeyHash> states_;

  /** \brief Block size of the related linear system */
  unsigned int num_block_entries_ = 0;
};

}  // namespace steam
