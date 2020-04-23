//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamCATrajPoseInterpEval.cpp
///
/// \author Tim Tang, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory_ca/SteamCATrajPoseInterpEval.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamCATrajPoseInterpEval::SteamCATrajPoseInterpEval(const Time& time,
                                   const SteamTrajVar::ConstPtr& knot1,
                                   const SteamTrajVar::ConstPtr& knot2) :
  knot1_(knot1), knot2_(knot2) {

  // Calculate time constants
  double t1 = knot1->getTime().seconds();
  double t2 = knot2->getTime().seconds();
  double tau = time.seconds();

  // Cheat by calculating deltas wrt t1, so we can avoid super large values
  tau = tau-t1;
  t2 = t2-t1;
  t1 = 0;
  
  double T = (knot2->getTime() - knot1->getTime()).seconds();
  double delta_tau = (time - knot1->getTime()).seconds();
  double delta_kappa = (knot2->getTime()-time).seconds();

  // std::cout << t1 << " " << t2 << " " << tau << std::endl;

  double T2 = T*T;
  double T3 = T2*T;
  double T4 = T3*T;
  double T5 = T4*T;

  double delta_tau2 = delta_tau*delta_tau;
  double delta_tau3 = delta_tau2*delta_tau;
  double delta_kappa2 = delta_kappa*delta_kappa;
  double delta_kappa3 = delta_kappa2*delta_kappa;

  // Calculate 'omega' interpolation values
  omega11_ = delta_tau3/T5*(t1*t1 - 5*t1*t2 + 3*t1*tau + 10*t2*t2 - 15*t2*tau + 6*tau*tau);
  omega12_ = delta_tau3*delta_kappa/T4*(t1 - 4*t2 + 3*tau);
  omega13_ = delta_tau3*delta_kappa2/(2*T3);

  // Calculate 'lambda' interpolation values
  lambda12_ = delta_tau*delta_kappa3/T4*(t2 - 4*t1 + 3*tau);
  lambda13_ = delta_tau2*delta_kappa3/(2*T3);

  // std::cout << "CA interpolation!" << std::endl;
  // std::cout << "omega11_" << omega11_ << std::endl;
  // std::cout << "ratio: " << t1*t1 - 5*t1*t2 + 3*t1*tau + 10*t2*t2 - 15*t2*tau + 6*tau*tau << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
SteamCATrajPoseInterpEval::Ptr SteamCATrajPoseInterpEval::MakeShared(const Time& time,
                                                   const SteamTrajVar::ConstPtr& knot1,
                                                   const SteamTrajVar::ConstPtr& knot2) {
  return SteamCATrajPoseInterpEval::Ptr(new SteamCATrajPoseInterpEval(time, knot1, knot2));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool SteamCATrajPoseInterpEval::isActive() const {
  return knot1_->getPose()->isActive()  ||
         !knot1_->getVelocity()->isLocked() ||
         knot2_->getPose()->isActive()  ||
         !knot2_->getVelocity()->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds references (shared pointers) to active state variables to the map output
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajPoseInterpEval::getActiveStateVariables(
    std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const {

  knot1_->getPose()->getActiveStateVariables(outStates);
  knot2_->getPose()->getActiveStateVariables(outStates);
  if (!knot1_->getVelocity()->isLocked()) {
    (*outStates)[knot1_->getVelocity()->getKey().getID()] = knot1_->getVelocity();
  }
  if (!knot2_->getVelocity()->isLocked()) {
    (*outStates)[knot2_->getVelocity()->getKey().getID()] = knot2_->getVelocity();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation SteamCATrajPoseInterpEval::evaluate() const {

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = knot2_->getPose()->evaluate()/knot1_->getPose()->evaluate();

  // Get se3 algebra of relative matrix
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Intermediate variable
  Eigen::Matrix<double,6,6> varpicurl2 = lgmath::se3::curlyhat(J_21_inv*knot2_->getVelocity()->getValue());

  // Calculate interpolated relative se3 algebra
  Eigen::Matrix<double,6,1> xi_i1 = lambda12_*knot1_->getVelocity()->getValue() +
                                    lambda13_*knot1_->getAcceleration()->getValue() +
                                    omega11_*xi_21 +
                                    omega12_*J_21_inv*knot2_->getVelocity()->getValue()+
                                    omega13_*(-0.5*varpicurl2*knot2_->getVelocity()->getValue() + J_21_inv*knot2_->getAcceleration()->getValue());

  // Calculate interpolated relative transformation matrix
  lgmath::se3::Transformation T_i1(xi_i1);

  // Return `global' interpolated transform
  // std::cout << "Interpolated xi_i1: " << std::endl;
  // std::cout << xi_i1.transpose() << std::endl;

  // std::cout << "part1: " << lambda12_*knot1_->getVelocity()->getValue().transpose() << std::endl;
  // std::cout << "part2: " << lambda13_*knot1_->getAcceleration()->getValue().transpose() << std::endl;
  // std::cout << "part3: " << omega11_*xi_21.transpose() << std::endl;
  // std::cout << "part4: " << (omega12_*J_21_inv*knot2_->getVelocity()->getValue()).transpose() << std::endl;
  // std::cout << "part5: " << (omega13_*(-0.5*varpicurl2*knot2_->getVelocity()->getValue())).transpose() << std::endl;
  // std::cout << "part6: " << (omega13_*J_21_inv*knot2_->getAcceleration()->getValue()).transpose() << std::endl;
  return T_i1*knot1_->getPose()->evaluate();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* SteamCATrajPoseInterpEval::evaluateTree() const {

  // Evaluate sub-trees
  EvalTreeNode<lgmath::se3::Transformation>* transform1 = knot1_->getPose()->evaluateTree();
  EvalTreeNode<lgmath::se3::Transformation>* transform2 = knot2_->getPose()->evaluateTree();

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = transform2->getValue()/transform1->getValue();

  // Get se3 algebra of relative matrix
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Intermediate variable
  Eigen::Matrix<double,6,6> varpicurl2 = lgmath::se3::curlyhat(J_21_inv*knot2_->getVelocity()->getValue());
  
  // Calculate interpolated relative se3 algebra
  Eigen::Matrix<double,6,1> xi_i1 = lambda12_*knot1_->getVelocity()->getValue() +
                                    lambda13_*knot1_->getAcceleration()->getValue() +
                                    omega11_*xi_21 +
                                    omega12_*J_21_inv*knot2_->getVelocity()->getValue()+
                                    omega13_*(-0.5*varpicurl2*knot2_->getVelocity()->getValue() + J_21_inv*knot2_->getAcceleration()->getValue());

  // Calculate interpolated relative transformation matrix
  lgmath::se3::Transformation T_i1(xi_i1);

  // Interpolated relative transform - new root node (using pool memory)
  EvalTreeNode<lgmath::se3::Transformation>* root = EvalTreeNode<lgmath::se3::Transformation>::pool.getObj();
  root->setValue(T_i1*transform1->getValue());

  // Add children
  root->addChild(transform1);
  root->addChild(transform2);

  // Return new root node
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Implementation for Block Automatic Differentiation
//////////////////////////////////////////////////////////////////////////////////////////////
template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
void SteamCATrajPoseInterpEval::appendJacobiansImpl(
    const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const {

  // Cast back to transformations
  EvalTreeNode<lgmath::se3::Transformation>* transform1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));
  EvalTreeNode<lgmath::se3::Transformation>* transform2 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(1));

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = transform2->getValue()/transform1->getValue();

  // Get se3 algebra of relative matrix
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();

  // Calculate the 6x6 associated Jacobian
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Intermediate variable
  Eigen::Matrix<double,6,6> varpicurl2 = lgmath::se3::curlyhat(J_21_inv*knot2_->getVelocity()->getValue());
  Eigen::Matrix<double,6,6> varpicurl = lgmath::se3::curlyhat(knot2_->getVelocity()->getValue());

  // Calculate interpolated relative se3 algebra
  Eigen::Matrix<double,6,1> xi_i1 = lambda12_*knot1_->getVelocity()->getValue() +
                                    lambda13_*knot1_->getAcceleration()->getValue() +
                                    omega11_*xi_21 +
                                    omega12_*J_21_inv*knot2_->getVelocity()->getValue()+
                                    omega13_*(-0.5*varpicurl2*knot2_->getVelocity()->getValue() + J_21_inv*knot2_->getAcceleration()->getValue());

  // Calculate interpolated relative transformation matrix
  lgmath::se3::Transformation T_i1(xi_i1);

  // Calculate the 6x6 Jacobian associated with the interpolated relative transformation matrix
  Eigen::Matrix<double,6,6> J_i1 = lgmath::se3::vec2jac(xi_i1);

  // Check if evaluator is active
  if (this->isActive()) {

    // Pose Jacobians
    if (knot1_->getPose()->isActive() || knot2_->getPose()->isActive()) {

      // Precompute matrix
      Eigen::Matrix<double,6,6> w = omega11_*J_i1*J_21_inv +
        0.5*omega12_*J_i1*varpicurl*J_21_inv + 0.25*omega13_*J_i1*varpicurl*varpicurl*J_21_inv+
        0.5*omega13_*J_i1*lgmath::se3::curlyhat(knot2_->getAcceleration()->getValue())*J_21_inv;

      // Check if transform1 is active
      if (knot1_->getPose()->isActive()) {
        Eigen::Matrix<double,6,6> jacobian = (-1) * w * T_21.adjoint() + T_i1.adjoint();
        knot1_->getPose()->appendBlockAutomaticJacobians(lhs*jacobian, transform1, outJacobians);
      }

      // Get index of split between left and right-hand-side of Jacobians
      unsigned int hintIndex = outJacobians->size();

      // Check if transform2 is active
      if (knot2_->getPose()->isActive()) {
        knot2_->getPose()->appendBlockAutomaticJacobians(lhs*w, transform2, outJacobians);
      }

      // Merge jacobians
      Jacobian<LHS_DIM,MAX_STATE_SIZE>::merge(outJacobians, hintIndex);
    }

    // 6 x 6 Velocity Jacobian 1
    if(!knot1_->getVelocity()->isLocked()) {

      // Add Jacobian
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot1_->getVelocity()->getKey(), lhs*lambda12_*J_i1));
    }

    // 6 x 6 Velocity Jacobian 2
    if(!knot2_->getVelocity()->isLocked()) {

      // Add Jacobian
      Eigen::Matrix<double,6,6> jacobian = omega12_*J_i1*J_21_inv - 0.5*omega13_*J_i1*(varpicurl2 - varpicurl*J_21_inv);
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot2_->getVelocity()->getKey(), lhs*jacobian));
    }

    // 6 x 6 Acceleration Jacobian 1
    if(!knot1_->getAcceleration()->isLocked()) {

      // Add Jacobian
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot1_->getAcceleration()->getKey(), lhs*lambda13_*J_i1));
    }
      
    // 6 x 6 Acceleration Jacobian 2
    if(!knot2_->getAcceleration()->isLocked()) {

      // Add Jacobian
      Eigen::Matrix<double,6,6> jacobian = omega13_*J_i1*J_21_inv;
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot2_->getAcceleration()->getKey(), lhs*jacobian));
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::MatrixXd& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,1,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<1,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,2,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<2,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,3,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<3,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,4,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<4,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamCATrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,6,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<6,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

} // se3
} // steam
