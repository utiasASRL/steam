#include "steam/problem/state_vector.hpp"

#include <iostream>

#include "steam/blockmat/BlockVector.hpp"

namespace steam {

StateVector StateVector::clone() const {
  StateVector cloned;
  // map is copied to avoid re-hashing all the entries,
  cloned.states_ = states_;
  cloned.num_block_entries_ = num_block_entries_;
  // go through the entries and perform a deep copy of each state
  for (auto it = cloned.states_.begin(); it != cloned.states_.end(); ++it)
    it->second.state = it->second.state->clone();
  return cloned;
}

void StateVector::copyValues(const StateVector &other) {
  // Check state vector are the same size
  if (this->states_.empty() ||
      this->num_block_entries_ != other.num_block_entries_ ||
      this->states_.size() != other.states_.size()) {
    throw std::invalid_argument(
        "StateVector size was not the same in copyValues()");
  }

  // Iterate over the state vectors and perform a "deep" copy without allocation
  // new memory. Keeping the original pointers is important as they are shared
  // in other places, and we want to update the shared memory.
  for (auto it = states_.begin(); it != states_.end(); ++it) {
    // Find matching state by ID
    auto it_other = other.states_.find(it->second.state->key());

    // Check that matching state was found and has the same structure
    if (it_other == other.states_.end() ||
        it->second.state->key() != it_other->second.state->key() ||
        it->second.local_block_index != it_other->second.local_block_index) {
      throw std::runtime_error(
          "StateVector was missing an entry in copyValues(), "
          "or structure of StateVector did not match.");
    }

    // Copy
    it->second.state->setFromCopy(it_other->second.state);
  }
}

void StateVector::addStateVariable(const StateVarBase::Ptr &state) {
  // Verify that state is not locked
  if (state->locked())
    throw std::invalid_argument(
        "Tried to add locked state variable to "
        "an optimizable state vector");

  // Verify we don't already have this state
  const auto &key = state->key();
  if (hasStateVariable(key))
    throw std::runtime_error(
        "StateVector already contains the state being added.");

  // Create new container
  StateContainer new_entry;
  new_entry.state = state;  // copy the shared_ptr (increases ref count)
  new_entry.local_block_index = num_block_entries_;
  states_.emplace(key, new_entry);

  // Increment number of entries
  num_block_entries_++;
}

bool StateVector::hasStateVariable(const StateKey &key) const {
  return states_.find(key) != states_.end();
}

StateVarBase::ConstPtr StateVector::getStateVariable(
    const StateKey &key) const {
  // Find the StateContainer for key
  const auto it = states_.find(key);

  // Check that it was found
  if (it == states_.end())
    throw std::runtime_error(
        "State variable was not found in call to getStateVariable()");

  // Return state variable reference
  return it->second.state;
}

unsigned int StateVector::getNumberOfStates() const { return states_.size(); }

int StateVector::getStateBlockIndex(const StateKey &key) const {
  // Find the StateContainer for key
  const auto it = states_.find(key);

  // Check that the state exists in the state vector
  //  **Note the likely causes that this occurs:
  //      1)  A cost term includes a state that is not added to the problem
  //      2)  A cost term is not checking whether states are locked, and adding
  //      a Jacobian for a locked state variable
  if (it == states_.end()) {
    std::stringstream err;
    err << "Tried to find a state that does not exist in the state vector";
    throw std::runtime_error(err.str());
  }

  // Return block index
  return it->second.local_block_index;
}

std::vector<unsigned int> StateVector::getStateBlockSizes() const {
  // Init return
  std::vector<unsigned int> result;
  result.resize(states_.size());

  // Iterate over states and populate result
  for (auto it = states_.begin(); it != states_.end(); ++it) {
    // Check that the local block index is in a valid range
    if (it->second.local_block_index < 0 ||
        it->second.local_block_index >= (int)result.size()) {
      throw std::logic_error("local_block_index is not a valid range");
    }

    // Populate return vector
    result[it->second.local_block_index] = it->second.state->perturb_dim();
  }

  return result;
}

void StateVector::update(const Eigen::VectorXd &perturbation) {
  // Convert single vector to a block-vector of perturbations (checks sizes)
  BlockVector blk_perturb(getStateBlockSizes(), perturbation);

  // Iterate over states and update each
  for (auto it = states_.begin(); it != states_.end(); ++it) {
    // Check for valid index
    if (it->second.local_block_index < 0)
      throw std::runtime_error("local_block_index is not initialized");

    // Update state
    it->second.state->update(blk_perturb.at(it->second.local_block_index));
  }
}

}  // namespace steam
