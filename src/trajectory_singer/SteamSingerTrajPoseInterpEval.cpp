//////////////////////////////////////////////////////////////////////////////////////////////
/// \file SteamSingerTrajPoseInterpEval.cpp
///
/// \author Jeremy Wong, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory_singer/SteamSingerTrajPoseInterpEval.hpp>

#include <lgmath.hpp>
#include <unsupported/Eigen/MatrixFunctions>
namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
SteamSingerTrajPoseInterpEval::SteamSingerTrajPoseInterpEval(const Time& time,
                                   const SteamSingerTrajVar::ConstPtr& knot1,
                                   const SteamSingerTrajVar::ConstPtr& knot2,
                                   const Eigen::Matrix<double,6,6>& alpha) :
  knot1_(knot1), knot2_(knot2), alpha_(alpha) {

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

  

  Eigen::Matrix<double,18,18> Q_delta_tau = getQmatrix(delta_tau);

  // double dt=delta_tau;
  // double dt2=dt*dt;
  // double dt3=dt2*dt;
  // Eigen::Matrix<double,6,6> eye=Eigen::Matrix<double,6,6>::Identity();
  // Eigen::Matrix<double,6,6> expon2=(-2*dt*alpha_).exp();
  // Eigen::Matrix<double,6,6> expon=(-dt*alpha_).exp();
  

  // Q_delta_tau.block<6,6>(0,0) = alpha4_inv*(eye-expon2+2*alpha_*dt+(2.0/3.0)*alpha3*dt3-2*alpha2*dt2-4*alpha_*dt*expon);
  // Q_delta_tau.block<6,6>(6,6) = alpha2_inv*(4*expon-3*eye-expon2+2*alpha_*dt);
  // Q_delta_tau.block<6,6>(12,12) = (eye-expon2);;
  // Q_delta_tau.block<6,6>(6,0) = Q_delta_tau.block<6,6>(0,6) = alpha3_inv*(expon2+eye-2*expon+2*alpha_*dt*expon-2*alpha_*dt+alpha2*dt2);
  // Q_delta_tau.block<6,6>(12,0) = Q_delta_tau.block<6,6>(0,12) = alpha2_inv*(eye-expon2-2*alpha_*dt*expon);
  // Q_delta_tau.block<6,6>(12,6) = Q_delta_tau.block<6,6>(6,12) = alpha_inv*(expon2+eye-2*expon);

  Eigen::Matrix<double,18,18> Q_T = getQmatrix(T);

  // dt=T;
  // dt2=dt*dt;
  // dt3=dt2*dt;
  // expon2=(-2*dt*alpha_).exp();
  // expon=(-dt*alpha_).exp();
  

  // Q_T.block<6,6>(0,0) = alpha4_inv*(eye-expon2+2*alpha_*dt+(2.0/3.0)*alpha3*dt3-2*alpha2*dt2-4*alpha_*dt*expon);
  // Q_T.block<6,6>(6,6) = alpha2_inv*(4*expon-3*eye-expon2+2*alpha_*dt);
  // Q_T.block<6,6>(12,12) = (eye-expon2);;
  // Q_T.block<6,6>(6,0) = Q_T.block<6,6>(0,6) = alpha3_inv*(expon2+eye-2*expon+2*alpha_*dt*expon-2*alpha_*dt+alpha2*dt2);
  // Q_T.block<6,6>(12,0) = Q_T.block<6,6>(0,12) = alpha2_inv*(eye-expon2-2*alpha_*dt*expon);
  // Q_T.block<6,6>(12,6) = Q_T.block<6,6>(6,12) = alpha_inv*(expon2+eye-2*expon);

  Eigen::Matrix<double,18,18> trans_delta_kappa=getTranMatrix(delta_kappa);

  Eigen::Matrix<double,18,18> omega=Q_delta_tau*trans_delta_kappa*Q_T.inverse();

  Eigen::Matrix<double,18,18> trans_delta_tau=getTranMatrix(delta_tau);
  Eigen::Matrix<double,18,18> trans_T=getTranMatrix(T);

  Eigen::Matrix<double,18,18> lambda=trans_delta_tau-omega*trans_T;

  omega11_=omega.block<6,6>(0,0);
  omega12_=omega.block<6,6>(0,6);
  omega13_=omega.block<6,6>(0,12);
  lambda12_=lambda.block<6,6>(0,6);
  lambda13_=lambda.block<6,6>(0,12);
  // std::cout << t1 << " " << t2 << " " << tau << std::endl;

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
SteamSingerTrajPoseInterpEval::Ptr SteamSingerTrajPoseInterpEval::MakeShared(const Time& time,
                                                   const SteamSingerTrajVar::ConstPtr& knot1,
                                                   const SteamSingerTrajVar::ConstPtr& knot2,
                                                   const Eigen::Matrix<double,6,6>& alpha) {
  return SteamSingerTrajPoseInterpEval::Ptr(new SteamSingerTrajPoseInterpEval(time, knot1, knot2, alpha));
}

Eigen::Matrix<double,18,18> SteamSingerTrajPoseInterpEval::getQmatrix(const double& dt) {
  Eigen::Matrix<double,6,6> alpha2=alpha_*alpha_;
  Eigen::Matrix<double,6,6> alpha3=alpha2*alpha_;
  Eigen::Matrix<double,6,6> alpha_inv=alpha_.inverse();
  Eigen::Matrix<double,6,6> alpha2_inv=alpha_inv*alpha_inv;
  Eigen::Matrix<double,6,6> alpha3_inv=alpha2_inv*alpha_inv;
  Eigen::Matrix<double,6,6> alpha4_inv=alpha3_inv*alpha_inv;

  Eigen::Matrix<double,18,18> Q;

  double dt2=dt*dt;
  double dt3=dt2*dt;
  Eigen::Matrix<double,6,6> eye=Eigen::Matrix<double,6,6>::Identity();
  Eigen::Matrix<double,6,6> expon2=(-2*dt*alpha_).exp();
  Eigen::Matrix<double,6,6> expon=(-dt*alpha_).exp();
  

  Q.block<6,6>(0,0) = alpha4_inv*(eye-expon2+2*alpha_*dt+(2.0/3.0)*alpha3*dt3-2*alpha2*dt2-4*alpha_*dt*expon);
  Q.block<6,6>(6,6) = alpha2_inv*(4*expon-3*eye-expon2+2*alpha_*dt);
  Q.block<6,6>(12,12) = (eye-expon2);;
  Q.block<6,6>(6,0) = Q.block<6,6>(0,6) = alpha3_inv*(expon2+eye-2*expon+2*alpha_*dt*expon-2*alpha_*dt+alpha2*dt2);
  Q.block<6,6>(12,0) = Q.block<6,6>(0,12) = alpha2_inv*(eye-expon2-2*alpha_*dt*expon);
  Q.block<6,6>(12,6) = Q.block<6,6>(6,12) = alpha_inv*(expon2+eye-2*expon);

  return Q;
}

Eigen::Matrix<double,18,18> SteamSingerTrajPoseInterpEval::getTranMatrix(const double& dt) {
  Eigen::Matrix<double,6,6> alpha_inv=alpha_.inverse();
  Eigen::Matrix<double,6,6> alpha2_inv=alpha_inv*alpha_inv;
  Eigen::Matrix<double,6,6> expon=(-dt*alpha_).exp();
  Eigen::Matrix<double,18,18> phi=Eigen::Matrix<double,18,18>::Identity();
  Eigen::Matrix<double,6,6> eye=Eigen::Matrix<double,6,6>::Identity();
  phi.block<6,6>(0,6)=dt*eye;
  phi.block<6,6>(0,12)=(-eye+dt*alpha_+expon)*alpha2_inv;
  phi.block<6,6>(6,12)=(eye-expon)*alpha_inv;
  phi.block<6,6>(12,12)=expon;
  return phi;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool SteamSingerTrajPoseInterpEval::isActive() const {
  return knot1_->getPose()->isActive()  ||
         !knot1_->getVelocity()->isLocked() ||
         knot2_->getPose()->isActive()  ||
         !knot2_->getVelocity()->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds references (shared pointers) to active state variables to the map output
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajPoseInterpEval::getActiveStateVariables(
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
lgmath::se3::Transformation SteamSingerTrajPoseInterpEval::evaluate() const {

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
EvalTreeNode<lgmath::se3::Transformation>* SteamSingerTrajPoseInterpEval::evaluateTree() const {

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
void SteamSingerTrajPoseInterpEval::appendJacobiansImpl(
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
      Eigen::Matrix<double,6,6> w = J_i1*omega11_*J_21_inv +
        0.5*J_i1*omega12_*varpicurl*J_21_inv + 0.25*J_i1*omega13_*varpicurl*varpicurl*J_21_inv+
        0.5*J_i1*omega13_*lgmath::se3::curlyhat(knot2_->getAcceleration()->getValue())*J_21_inv;

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
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot1_->getVelocity()->getKey(), lhs*J_i1*lambda12_));
    }

    // 6 x 6 Velocity Jacobian 2
    if(!knot2_->getVelocity()->isLocked()) {

      // Add Jacobian
      Eigen::Matrix<double,6,6> jacobian = J_i1*omega12_*J_21_inv - 0.5*J_i1*omega13_*(varpicurl2 - varpicurl*J_21_inv);
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot2_->getVelocity()->getKey(), lhs*jacobian));
    }

    // 6 x 6 Acceleration Jacobian 1
    if(!knot1_->getAcceleration()->isLocked()) {

      // Add Jacobian
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot1_->getAcceleration()->getKey(), lhs*J_i1*lambda13_));
    }
      
    // 6 x 6 Acceleration Jacobian 2
    if(!knot2_->getAcceleration()->isLocked()) {

      // Add Jacobian
      Eigen::Matrix<double,6,6> jacobian = J_i1*omega13_*J_21_inv;
      outJacobians->push_back(Jacobian<LHS_DIM,MAX_STATE_SIZE>(knot2_->getAcceleration()->getKey(), lhs*jacobian));
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::MatrixXd& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,1,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<1,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,2,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<2,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,3,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<3,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,4,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<4,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void SteamSingerTrajPoseInterpEval::appendBlockAutomaticJacobians(
    const Eigen::Matrix<double,6,6>& lhs,
    EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
    std::vector<Jacobian<6,6> >* outJacobians) const {
  this->appendJacobiansImpl(lhs,evaluationTree, outJacobians);
}

} // se3
} // steam
