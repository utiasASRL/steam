//////////////////////////////////////////////////////////////////////////////////////////////
/// \file GpTrajectoryPrior.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/GpTrajectoryPrior.hpp>

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
  return knot1_->T_k_root->isActive()  ||
         !knot1_->varpi->isLocked() ||
         knot2_->T_k_root->isActive()  ||
         !knot2_->varpi->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the GP prior factor
//////////////////////////////////////////////////////////////////////////////////////////////
//Eigen::Matrix<double,12,1> GpTrajectoryPrior::evaluate() const {
Eigen::VectorXd GpTrajectoryPrior::evaluate() const {

  // Precompute values
  lgmath::se3::Transformation T_21 = knot2_->T_k_root->evaluate()/knot1_->T_k_root->evaluate();
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
//Eigen::Matrix<double,12,1> GpTrajectoryPrior::evaluate(const Eigen::Matrix<double,12,12>& lhs,
//                                                       std::vector<Jacobian<12,6> >* jacs) const {
Eigen::VectorXd GpTrajectoryPrior::evaluate(const Eigen::MatrixXd& lhs,
                                            std::vector<Jacobian<> >* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation trees
  EvalTreeNode<lgmath::se3::Transformation>* evaluationTree1 = knot1_->T_k_root->evaluateTree();
  EvalTreeNode<lgmath::se3::Transformation>* evaluationTree2 = knot2_->T_k_root->evaluateTree();

  // Get evaluations (from trees)
  lgmath::se3::Transformation T_21 = evaluationTree2->getValue()/evaluationTree1->getValue();
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  double deltaTime = (knot2_->time - knot1_->time).seconds();

  // Knot 1 transform
  if(knot1_->T_k_root->isActive()) {
    Eigen::Matrix<double,6,6> Jinv_12 = J_21_inv*T_21.adjoint();

    // Construct jacobian
    Eigen::Matrix<double,12,6> jacobian;
    jacobian.topRows<6>() = -Jinv_12;
    jacobian.bottomRows<6>() = -0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*Jinv_12;

    // Get Jacobians
    knot1_->T_k_root->appendJacobians(lhs * jacobian, evaluationTree1, jacs);
  }

  // Get index of split between left and right-hand-side of Jacobians
  unsigned int hintIndex = jacs->size();

  // Knot 2 transform
  if(knot2_->T_k_root->isActive()) {

    // Construct jacobian
    Eigen::Matrix<double,12,6> jacobian;
    jacobian.topRows<6>() = J_21_inv;
    jacobian.bottomRows<6>() = 0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*J_21_inv;

    // Get Jacobians
    knot2_->T_k_root->appendJacobians(lhs * jacobian, evaluationTree2, jacs);
  }

  // Merge jacobians
  Jacobian<>::merge(jacs, hintIndex);

  // Knot 1 velocity
  if(!knot1_->varpi->isLocked()) {

    // Construct Jacobian Object
    jacs->push_back(Jacobian<>());
    Jacobian<>& jacref = jacs->back();
    jacref.key = knot1_->varpi->getKey();

    // Fill in matrix
    Eigen::Matrix<double,12,6> jacobian;
    jacobian.topRows<6>() = -deltaTime*Eigen::Matrix<double,6,6>::Identity();
    jacobian.bottomRows<6>() = -Eigen::Matrix<double,6,6>::Identity();
    jacref.jac = lhs * jacobian;
  }

  // Knot 2 velocity
  if(!knot2_->varpi->isLocked()) {

    // Construct Jacobian Object
    jacs->push_back(Jacobian<>());
    Jacobian<>& jacref = jacs->back();
    jacref.key = knot2_->varpi->getKey();

    // Fill in matrix
    Eigen::Matrix<double,12,6> jacobian;
    jacobian.topRows<6>() = Eigen::Matrix<double,6,6>::Zero();
    jacobian.bottomRows<6>() = J_21_inv;
    jacref.jac = lhs * jacobian;
  }

  // Return tree memory to pool
  EvalTreeNode<lgmath::se3::Transformation>::pool.returnObj(evaluationTree1);
  EvalTreeNode<lgmath::se3::Transformation>::pool.returnObj(evaluationTree2);

  // Return error
  Eigen::Matrix<double,12,1> error;
  error.head<6>() = xi_21 - deltaTime*knot1_->varpi->getValue();
  error.tail<6>() = J_21_inv * knot2_->varpi->getValue() - knot1_->varpi->getValue();
  return error;

  // Precompute values
//  lgmath::se3::Transformation T_21 = knot2_->T_k_root->evaluate()/knot1_->T_k_root->evaluate();
//  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
//  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
//  double deltaTime = (knot2_->time - knot1_->time).seconds();

//  // Check and initialize jacobian array
//  if (jacs == NULL) {
//    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
//  }
//  jacs->clear();
//  jacs->reserve(4);

//  // Knot 1 transform
//  if(knot1_->T_k_root->isActive()) {
//    Eigen::Matrix<double,6,6> Jinv_12 = J_21_inv*T_21.adjoint();

//    // Construct Jacobian Object
//    jacs->push_back(Jacobian<12,6>());
//    Jacobian<12,6>& jacref = jacs->back();
//    jacref.key = knot1_->T_k0->getKey();

//    // Fill in matrix
//    Eigen::Matrix<double,12,6> jacobian;
//    jacobian.topRows<6>() = -Jinv_12;
//    jacobian.bottomRows<6>() = -0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*Jinv_12;
//    jacref.jac = lhs * jacobian;
//  }

//  // Knot 1 velocity
//  if(!knot1_->varpi->isLocked()) {

//    // Construct Jacobian Object
//    jacs->push_back(Jacobian<12,6>());
//    Jacobian<12,6>& jacref = jacs->back();
//    jacref.key = knot1_->varpi->getKey();

//    // Fill in matrix
//    Eigen::Matrix<double,12,6> jacobian;
//    jacobian.topRows<6>() = -deltaTime*Eigen::Matrix<double,6,6>::Identity();
//    jacobian.bottomRows<6>() = -Eigen::Matrix<double,6,6>::Identity();
//    jacref.jac = lhs * jacobian;
//  }

//  // Knot 2 transform
//  if(knot2_->T_k_root->isActive()) {

//    // Construct Jacobian Object
//    jacs->push_back(Jacobian<12,6>());
//    Jacobian<12,6>& jacref = jacs->back();
//    jacref.key = knot2_->T_k0->getKey();

//    // Fill in matrix
//    Eigen::Matrix<double,12,6> jacobian;
//    jacobian.topRows<6>() = J_21_inv;
//    jacobian.bottomRows<6>() = 0.5*lgmath::se3::curlyhat(knot2_->varpi->getValue())*J_21_inv;
//    jacref.jac = lhs * jacobian;
//  }

//  // Knot 2 velocity
//  if(!knot2_->varpi->isLocked()) {

//    // Construct Jacobian Object
//    jacs->push_back(Jacobian<12,6>());
//    Jacobian<12,6>& jacref = jacs->back();
//    jacref.key = knot2_->varpi->getKey();

//    // Fill in matrix
//    Eigen::Matrix<double,12,6> jacobian;
//    jacobian.topRows<6>() = Eigen::Matrix<double,6,6>::Zero();
//    jacobian.bottomRows<6>() = J_21_inv;
//    jacref.jac = lhs * jacobian;
//  }

//  // Return error
//  Eigen::Matrix<double,12,1> error;
//  error.head<6>() = xi_21 - deltaTime*knot1_->varpi->getValue();
//  error.tail<6>() = J_21_inv * knot2_->varpi->getValue() - knot1_->varpi->getValue();
//  return error;
}

} // se3
} // steam
