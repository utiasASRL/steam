//////////////////////////////////////////////////////////////////////////////////////////////
/// \file StateVector.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/StateVector.hpp>
#include <steam/sparse/BlockVector.hpp>

#include <iostream>
#include <glog/logging.h>


namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor
//////////////////////////////////////////////////////////////////////////////////////////////
StateVector::StateVector() : numBlockEntries_(0) {}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Copy constructor -- deep copy
//////////////////////////////////////////////////////////////////////////////////////////////
StateVector::StateVector(const StateVector& other) : states_(other.states_),
  numBlockEntries_(other.numBlockEntries_) {

  // Map is copied in initialization list to avoid re-hashing all the entries,
  // now we go through the entries and perform a deep copy
  boost::unordered_map<StateID, StateContainer>::iterator it = states_.begin();
  for(; it != states_.end(); ++it) {
    it->second.state = it->second.state->clone();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Assignment operator -- deep copy
//////////////////////////////////////////////////////////////////////////////////////////////
StateVector& StateVector::operator= (const StateVector& other) {

  // Copy-swap idiom
  StateVector tmp(other); // note, copy constructor makes a deep copy
  std::swap(states_, tmp.states_);
  std::swap(numBlockEntries_, tmp.numBlockEntries_);
  return *this;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Copy the values of 'other' into 'this' (states must already align, typically
///        this means that one is already a deep copy of the other)
//////////////////////////////////////////////////////////////////////////////////////////////
void StateVector::copyValues(const StateVector& other) {

  // Check state vector are the same size
  CHECK(!states_.empty());
  CHECK(this->numBlockEntries_ == other.numBlockEntries_);
  CHECK(this->states_.size() == other.states_.size());

  // Iterate over the state vectors and perform a "deep" copy without allocation new memory.
  // Keeping the original pointers is important as they are shared in other places, and we
  // want to update the shared memory.
  // todo: can we avoid a 'find' here?
  boost::unordered_map<StateID, StateContainer>::iterator it = states_.begin();
  for(; it != states_.end(); ++it) {

    // Find matching state by ID
    boost::unordered_map<StateID, StateContainer>::const_iterator itOther = other.states_.find(it->second.state->getKey().getID());
    CHECK(itOther != other.states_.end());

    // Check state variables and state structure are the same
    CHECK(it->second.state->getKey().getID() == itOther->second.state->getKey().getID()) << ", failed: " << it->second.state->getKey().getID() << " == " << itOther->second.state->getKey().getID();
    CHECK(it->second.localBlockIndex == itOther->second.localBlockIndex) << ", failed: " << it->second.localBlockIndex << " == " << itOther->second.localBlockIndex;

    // Copy
    it->second.state->setFromCopy(itOther->second.state);
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add state variable
//////////////////////////////////////////////////////////////////////////////////////////////
void StateVector::addStateVariable(const StateVariableBase::Ptr& state) {

  // Verify that state is not locked
  CHECK(!state->isLocked()) << "Tried to add locked state variable to an optimizable state vector.";

  // Verify we don't already have this state
  StateKey key = state->getKey();
  CHECK(!this->hasStateVariable(key));

  // Create new container
  StateContainer newEntry;
  newEntry.state = state; // copy the shared_ptr (increases ref count)
  newEntry.localBlockIndex = numBlockEntries_;
  states_[key.getID()] = newEntry;

  // Increment number of entries
  numBlockEntries_++;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Check if a state variable exists in the vector
//////////////////////////////////////////////////////////////////////////////////////////////
bool StateVector::hasStateVariable(const StateKey& key) const {

  // Find the StateContainer for key
  boost::unordered_map<StateID, StateContainer>::const_iterator it = states_.find(key.getID());

  // Return if found
  return it != states_.end();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get a state variable using a key
//////////////////////////////////////////////////////////////////////////////////////////////
StateVariableBase::ConstPtr StateVector::getStateVariable(const StateKey& key) const {

  // Find the StateContainer for key
  boost::unordered_map<StateID, StateContainer>::const_iterator it = states_.find(key.getID());

  // Check that it was found
  CHECK(it != states_.end());

  // Return state variable reference
  return it->second.state;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get number of state variables
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int StateVector::getNumberOfStates() const {
  return states_.size();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the block index of a state
//////////////////////////////////////////////////////////////////////////////////////////////
int StateVector::getStateBlockIndex(const StateKey& key) const {

  // Find the StateContainer for key
  boost::unordered_map<StateID, StateContainer>::const_iterator it = states_.find(key.getID());

  // Check that the state exists in the state vector
  //  **Note the likely causes that this occurs:
  //      1)  A cost term includes a state that is not added to the problem
  //      2)  A cost term is not checking whether states are locked, and adding a Jacobian for a locked state variable
  CHECK(it != states_.end()) << ", tried to find a state that does not exist in the state vector (ID: " << key.getID() << ").";

  // Return block index
  return it->second.localBlockIndex;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get an ordered list of the sizes of the 'block' state variables
//////////////////////////////////////////////////////////////////////////////////////////////
std::vector<unsigned int> StateVector::getStateBlockSizes() const {

  // Init return
  std::vector<unsigned int> result;
  result.resize(states_.size());

  // Iterate over states and populate result
  for (boost::unordered_map<StateID, StateContainer>::const_iterator it = states_.begin();
       it != states_.end(); ++it ) {

    // Check that the local block index is in a valid range
    CHECK(it->second.localBlockIndex >= 0 && it->second.localBlockIndex < (int)result.size());

    // Populate return vector
    result[it->second.localBlockIndex] = it->second.state->getPerturbDim();
  }

  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Update the state vector
//////////////////////////////////////////////////////////////////////////////////////////////
void StateVector::update(const Eigen::VectorXd& perturbation) {

  // Convert single vector to a block-vector of perturbations (this checks sizes)
  BlockVector blkPerturb(this->getStateBlockSizes(), perturbation);

  // Iterate over states and update each
  for ( boost::unordered_map<StateID, StateContainer>::const_iterator it = states_.begin(); it != states_.end(); ++it ) {

    // Check for valid index
    CHECK(it->second.localBlockIndex >= 0);

    // Update state
    it->second.state->update(blkPerturb.getBlkVector(it->second.localBlockIndex));
  }
}

} // steam
