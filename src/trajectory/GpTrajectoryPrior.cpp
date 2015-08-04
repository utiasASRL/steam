//////////////////////////////////////////////////////////////////////////////////////////////
/// \file GpTrajectoryPrior.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/GpTrajectoryPrior.hpp>

#include <steam/evaluator/jacobian/JacobianTreeBranchNode.hpp>
#include <steam/evaluator/jacobian/JacobianTreeLeafNode.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
GpTrajectoryPrior::GpTrajectoryPrior(const GpTrajectory::Knot::ConstPtr& knot1,
                                     const GpTrajectory::Knot::ConstPtr& knot2) :
  knot1_(knot1), knot2_(knot2) {
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool GpTrajectoryPrior::isActive() const {
  return !knot1_->T_k0->isLocked()  ||
         !knot1_->varpi->isLocked() ||
         !knot2_->T_k0->isLocked()  ||
         !knot2_->varpi->isLocked();
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the GP prior factor
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GpTrajectoryPrior::evaluate() const {

  // Precompute values
  lgmath::se3::Transformation T_21 = knot2_->T_k0->getValue()/knot1_->T_k0->getValue();
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Compute error
  Eigen::Matrix<double,12,1> error;
  error.head<6>() = xi_21 - (knot2_->time - knot1_->time).seconds()*knot1_->varpi->getValue();
  error.tail<6>() = J_21_inv * knot2_->varpi->getValue() - knot1_->varpi->getValue();
  return error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the GP prior factor and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXd GpTrajectoryPrior::evaluate(std::vector<Jacobian>* jacs) const {

  // Precompute values
  lgmath::se3::Transformation T_21 = knot2_->T_k0->getValue()/knot1_->T_k0->getValue();
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  double deltaTime = (knot2_->time - knot1_->time).seconds();

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();
  jacs->reserve(4);

  if(!knot1_->T_k0->isLocked()) {
    Eigen::Matrix<double,6,6> Jinv_12 = J_21_inv*T_21.adjoint();

    jacs->push_back(Jacobian());
    Jacobian& jacref = jacs->back();
    jacref.key = knot1_->T_k0->getKey();
    jacref.jac = Eigen::Matrix<double,12,6>();
    jacref.jac.block<6,6>(0,0) = -Jinv_12;
    jacref.jac.block<6,6>(6,0) = -0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*Jinv_12;
  }

  if(!knot1_->varpi->isLocked()) {
    jacs->push_back(Jacobian());
    Jacobian& jacref = jacs->back();
    jacref.key = knot1_->varpi->getKey();
    jacref.jac = Eigen::Matrix<double,12,6>();
    jacref.jac.block<6,6>(0,0) = -deltaTime*Eigen::Matrix<double,6,6>::Identity();
    jacref.jac.block<6,6>(6,0) = -Eigen::Matrix<double,6,6>::Identity();
  }

  if(!knot2_->T_k0->isLocked()) {
    jacs->push_back(Jacobian());
    Jacobian& jacref = jacs->back();
    jacref.key = knot2_->T_k0->getKey();
    jacref.jac = Eigen::Matrix<double,12,6>();
    jacref.jac.block<6,6>(0,0) = J_21_inv;
    jacref.jac.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*J_21_inv;
  }

  if(!knot2_->varpi->isLocked()) {
    jacs->push_back(Jacobian());
    Jacobian& jacref = jacs->back();
    jacref.key = knot2_->varpi->getKey();
    jacref.jac = Eigen::Matrix<double,12,6>();
    jacref.jac.block<6,6>(0,0) = Eigen::Matrix<double,6,6>::Zero();
    jacref.jac.block<6,6>(6,0) = J_21_inv;
  }

  // Return error
  Eigen::Matrix<double,12,1> error;
  error.head<6>() = xi_21 - deltaTime*knot1_->varpi->getValue();
  error.tail<6>() = J_21_inv * knot2_->varpi->getValue() - knot1_->varpi->getValue();
  return error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the GP prior factor and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
std::pair<Eigen::VectorXd, JacobianTreeNode::ConstPtr> GpTrajectoryPrior::evaluateJacobians() const {

  // Init Jacobian node (null)
  JacobianTreeBranchNode::Ptr jacobianNode;

  // Precompute values
  lgmath::se3::Transformation T_21 = knot2_->T_k0->getValue()/knot1_->T_k0->getValue();
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  double deltaTime = (knot2_->time - knot1_->time).seconds();

  // Check if evaluator is active
  if (this->isActive()) {

    // Make Jacobian node
    jacobianNode = JacobianTreeBranchNode::Ptr(new JacobianTreeBranchNode());

    // Knot 1 transform
    if(!knot1_->T_k0->isLocked()) {
      Eigen::Matrix<double,6,6> Jinv_12 = J_21_inv*T_21.adjoint();

      // Make leaf node for Landmark
      JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot1_->T_k0));

      // Construct Jacobian
      Eigen::Matrix<double,12,6> jacobian;
      jacobian.block<6,6>(0,0) = -Jinv_12;
      jacobian.block<6,6>(6,0) = -0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*Jinv_12;

      // Add Jacobian
      jacobianNode->add(jacobian, leafNode);
    }

    // Knot 1 velocity
    if(!knot1_->varpi->isLocked()) {

      // Make leaf node for Landmark
      JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot1_->varpi));

      // Construct Jacobian
      Eigen::Matrix<double,12,6> jacobian;
      jacobian.block<6,6>(0,0) = -deltaTime*Eigen::Matrix<double,6,6>::Identity();
      jacobian.block<6,6>(6,0) = -Eigen::Matrix<double,6,6>::Identity();

      // Add Jacobian
      jacobianNode->add(jacobian, leafNode);
    }

    // Knot 2 transform
    if(!knot2_->T_k0->isLocked()) {
      Eigen::Matrix<double,6,6> Jinv_12 = J_21_inv*T_21.adjoint();

      // Make leaf node for Landmark
      JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot2_->T_k0));

      // Construct Jacobian
      Eigen::Matrix<double,12,6> jacobian;
      jacobian.block<6,6>(0,0) = J_21_inv;
      jacobian.block<6,6>(6,0) = 0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*J_21_inv;

      // Add Jacobian
      jacobianNode->add(jacobian, leafNode);
    }

    // Knot 2 velocity
    if(!knot2_->varpi->isLocked()) {

      // Make leaf node for Landmark
      JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot2_->varpi));

      // Construct Jacobian
      Eigen::Matrix<double,12,6> jacobian;
      jacobian.block<6,6>(0,0) = Eigen::Matrix<double,6,6>::Zero();
      jacobian.block<6,6>(6,0) = J_21_inv;

      // Add Jacobian
      jacobianNode->add(jacobian, leafNode);
    }
  }

  // Return error
  Eigen::Matrix<double,12,1> error;
  error.head<6>() = xi_21 - deltaTime*knot1_->varpi->getValue();
  error.tail<6>() = J_21_inv * knot2_->varpi->getValue() - knot1_->varpi->getValue();
  return std::make_pair(error, jacobianNode);
}

} // se3
} // steam
