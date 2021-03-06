//////////////////////////////////////////////////////////////////////////////////////////////
/// \file VectorSpaceStateVar.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/state/VectorSpaceStateVar.hpp>

namespace steam {

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Construct from Eigen vector (perturbation dimension assumed to match vector)
/////////////////////////////////////////////////////////////////////////////////////////////
VectorSpaceStateVar::VectorSpaceStateVar(Eigen::VectorXd v) : StateVariable(v, v.size()) {}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Update the vector space state variable via an additive perturbation:
///          this += perturbation
/////////////////////////////////////////////////////////////////////////////////////////////
bool VectorSpaceStateVar::update(const Eigen::VectorXd& perturbation) {

  if (perturbation.size() != this->getPerturbDim()) {
    throw std::runtime_error("During attempt to update a state variable, the provided "
                             "perturbation (VectorXd) was not the correct size.");
  }

  this->value_ = this->value_ + perturbation;
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Clone method
/////////////////////////////////////////////////////////////////////////////////////////////
StateVariableBase::Ptr VectorSpaceStateVar::clone() const {
  return VectorSpaceStateVar::Ptr(new VectorSpaceStateVar(*this));
}


} // steam
