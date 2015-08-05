//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvaluators.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/trajectory/GpTrajectoryEval.hpp>

#include <steam/evaluator/jacobian/JacobianTreeBranchNode.hpp>
#include <steam/evaluator/jacobian/JacobianTreeLeafNode.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
GpTrajectoryEval::GpTrajectoryEval(const Time& time, const GpTrajectory::Knot::ConstPtr& knot1,
                                   const GpTrajectory::Knot::ConstPtr& knot2) :
  knot1_(knot1), knot2_(knot2) {

  double tau = (time - knot1->time).seconds();
  double T = (knot2->time - knot1->time).seconds();
  double ratio = tau/T;
  double ratio2 = ratio*ratio;
  double ratio3 = ratio2*ratio;

  psi11_ = 3*ratio2 - 2*ratio3;
  psi12_ = tau*(ratio2 - ratio);
  psi21_ = 6*(ratio - ratio2)/T;
  psi22_ = 3*ratio2 - 2*ratio;

  lambda11_ = 1 - psi11_;
  lambda12_ = tau - T*psi11_ - psi12_;
  lambda21_ = -psi21_;
  lambda22_ = 1 - psi21_ - psi22_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
GpTrajectoryEval::Ptr GpTrajectoryEval::MakeShared(const Time& time, const GpTrajectory::Knot::ConstPtr& knot1,
                                                   const GpTrajectory::Knot::ConstPtr& knot2) {
  return GpTrajectoryEval::Ptr(new GpTrajectoryEval(time, knot1, knot2));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool GpTrajectoryEval::isActive() const {
  return !knot1_->T_k0->isLocked()  ||
         !knot1_->varpi->isLocked() ||
         !knot2_->T_k0->isLocked()  ||
         !knot2_->varpi->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation GpTrajectoryEval::evaluate() const {

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = knot2_->T_k0->getValue()/knot1_->T_k0->getValue();
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Interpolated relative transform
  Eigen::Matrix<double,6,1> xi_i1 = lambda12_*knot1_->varpi->getValue() +
                                    psi11_*xi_21 +
                                    psi12_*J_21_inv*knot2_->varpi->getValue();
  lgmath::se3::Transformation T_i1(xi_i1);

  // Return `global' interpolated transform
  return T_i1*knot1_->T_k0->getValue();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix and Jacobian (identity)
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation GpTrajectoryEval::evaluate(std::vector<Jacobian>* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();
  jacs->reserve(4);

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = knot2_->T_k0->getValue()/knot1_->T_k0->getValue();
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Interpolated relative transform
  Eigen::Matrix<double,6,1> xi_i1 = lambda12_*knot1_->varpi->getValue() +
                                    psi11_*xi_21 +
                                    psi12_*J_21_inv*knot2_->varpi->getValue();
  lgmath::se3::Transformation T_i1(xi_i1);
  Eigen::Matrix<double,6,6> J_i1 = lgmath::se3::vec2jac(xi_i1);

  // Pose Jacobians
  if (!knot1_->T_k0->isLocked() || !knot2_->T_k0->isLocked()) {

    Eigen::Matrix<double,6,6> w = psi11_*J_i1*J_21_inv +
      0.5*psi12_*J_i1*lgmath::se3::curlyhat(knot2_->varpi->getValue())*J_21_inv;

    // 6 x 6 Pose Jacobian 1
    if(!knot1_->T_k0->isLocked()) {
      jacs->push_back(Jacobian());
      Jacobian& jacref = jacs->back();
      jacref.key = knot1_->T_k0->getKey();
      jacref.jac = -w*T_21.adjoint() + T_i1.adjoint();
    }

    // 6 x 6 Pose Jacobian 2
    if(!knot2_->T_k0->isLocked()) {
      jacs->push_back(Jacobian());
      Jacobian& jacref = jacs->back();
      jacref.key = knot2_->T_k0->getKey();
      jacref.jac = w;
    }
  }

  // 6 x 6 Velocity Jacobian 1
  if(!knot1_->varpi->isLocked()) {
    jacs->push_back(Jacobian());
    Jacobian& jacref = jacs->back();
    jacref.key = knot1_->varpi->getKey();
    jacref.jac = lambda12_*J_i1;
  }

  // 6 x 6 Velocity Jacobian 2
  if(!knot2_->varpi->isLocked()) {
    jacs->push_back(Jacobian());
    Jacobian& jacref = jacs->back();
    jacref.key = knot2_->varpi->getKey();
    jacref.jac = psi12_*J_i1*J_21_inv;;
  }

  // Return `global' interpolated transform
  return T_i1*knot1_->T_k0->getValue();

}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix and Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
std::pair<lgmath::se3::Transformation, JacobianTreeNode::ConstPtr> GpTrajectoryEval::evaluateJacobians() const {

  // Init Jacobian node (null)
  JacobianTreeBranchNode::Ptr jacobianNode;

  // Get relative matrix info
  lgmath::se3::Transformation T_21 = knot2_->T_k0->getValue()/knot1_->T_k0->getValue();
  Eigen::Matrix<double,6,1> xi_21 = T_21.vec();
  Eigen::Matrix<double,6,6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);

  // Interpolated relative transform
  Eigen::Matrix<double,6,1> xi_i1 = lambda12_*knot1_->varpi->getValue() +
                                    psi11_*xi_21 +
                                    psi12_*J_21_inv*knot2_->varpi->getValue();
  lgmath::se3::Transformation T_i1(xi_i1);
  Eigen::Matrix<double,6,6> J_i1 = lgmath::se3::vec2jac(xi_i1);

  // Check if evaluator is active
  if (this->isActive()) {

    // Make Jacobian node
    jacobianNode = JacobianTreeBranchNode::Ptr(new JacobianTreeBranchNode(4));

    // Pose Jacobians
    if (!knot1_->T_k0->isLocked() || !knot2_->T_k0->isLocked()) {

      // Precompute matrix
      Eigen::Matrix<double,6,6> w = psi11_*J_i1*J_21_inv +
        0.5*psi12_*J_i1*lgmath::se3::curlyhat(knot2_->varpi->getValue())*J_21_inv;

      // 6 x 6 Pose Jacobian 1
      if(!knot1_->T_k0->isLocked()) {

        // Make leaf node for Landmark
        JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot1_->T_k0));

        // Add Jacobian
        //jacobianNode->add(-w*T_21.adjoint() + T_i1.adjoint(), leafNode);
        jacobianNode->add(leafNode) = -w*T_21.adjoint() + T_i1.adjoint();
      }

      // 6 x 6 Pose Jacobian 2
      if(!knot2_->T_k0->isLocked()) {

        // Make leaf node for Landmark
        JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot2_->T_k0));

        // Add Jacobian
        //jacobianNode->add(w, leafNode);
        jacobianNode->add(leafNode) = w;
      }
    }

    // 6 x 6 Velocity Jacobian 1
    if(!knot1_->varpi->isLocked()) {

      // Make leaf node for Landmark
      JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot1_->varpi));

      // Add Jacobian
      //jacobianNode->add(lambda12_*J_i1, leafNode);
      jacobianNode->add(leafNode) = lambda12_*J_i1;
    }

    // 6 x 6 Velocity Jacobian 2
    if(!knot2_->varpi->isLocked()) {

      // Make leaf node for Landmark
      JacobianTreeLeafNode::Ptr leafNode(new JacobianTreeLeafNode(knot2_->varpi));

      // Add Jacobian
      //jacobianNode->add(psi12_*J_i1*J_21_inv, leafNode);
      jacobianNode->add(leafNode) = psi12_*J_i1*J_21_inv;
    }
  }

  // Return `global' interpolated transform
  return std::make_pair(T_i1*knot1_->T_k0->getValue(), jacobianNode);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* GpTrajectoryEval::evaluateTree() const {

  // Precalculate and store linearization values for later use in appendJacobians

  // Get relative matrix info
  EvalTreeNode<lgmath::se3::Transformation>* T_21 =
      new EvalTreeNode<lgmath::se3::Transformation>(knot2_->T_k0->getValue()/knot1_->T_k0->getValue());

  Eigen::Matrix<double,6,1> xi_21 = T_21->getValue().vec();

  EvalTreeNode<Eigen::Matrix<double,6,6> >* J_21_inv =
      new EvalTreeNode<Eigen::Matrix<double,6,6> >(lgmath::se3::vec2jacinv(xi_21));

  // Interpolated relative transform
  EvalTreeNode<Eigen::Matrix<double,6,1> >* xi_i1 =
      new EvalTreeNode<Eigen::Matrix<double,6,1> >(lambda12_*knot1_->varpi->getValue() +
                                                   psi11_*xi_21 +
                                                   psi12_*J_21_inv->getValue()*knot2_->varpi->getValue());

  EvalTreeNode<lgmath::se3::Transformation>* T_i1 =
      new EvalTreeNode<lgmath::se3::Transformation>(lgmath::se3::Transformation(xi_i1->getValue()));

  // Interpolated relative transform - new root node
  EvalTreeNode<lgmath::se3::Transformation>* root =
      new EvalTreeNode<lgmath::se3::Transformation>(T_i1->getValue()*knot1_->T_k0->getValue());

  // Add children - for later use
  root->addChild(T_21);
  root->addChild(J_21_inv);
  root->addChild(xi_i1);
  root->addChild(T_i1);

  // Return `global' interpolated transform
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void GpTrajectoryEval::appendJacobians(const Eigen::MatrixXd& lhs,
                             EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                             std::vector<Jacobian>* outJacobians) const {

  // Get precalculated linearization values

  // Get relative matrix info
  EvalTreeNode<lgmath::se3::Transformation>* T_21 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  EvalTreeNode<Eigen::Matrix<double,6,6> >* J_21_inv =
      static_cast<EvalTreeNode<Eigen::Matrix<double,6,6> >*>(evaluationTree->childAt(1));

  // Interpolated relative transform
  EvalTreeNode<Eigen::Matrix<double,6,1> >* xi_i1 =
      static_cast<EvalTreeNode<Eigen::Matrix<double,6,1> >*>(evaluationTree->childAt(2));

  EvalTreeNode<lgmath::se3::Transformation>* T_i1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(3));

  // Calculate relative interp jacobian
  Eigen::Matrix<double,6,6> J_i1 = lgmath::se3::vec2jac(xi_i1->getValue());

  // Check if evaluator is active
  if (this->isActive()) {

    // Pose Jacobians
    if (!knot1_->T_k0->isLocked() || !knot2_->T_k0->isLocked()) {

      // Precompute matrix
      Eigen::Matrix<double,6,6> w = psi11_*J_i1*J_21_inv->getValue() +
        0.5*psi12_*J_i1*lgmath::se3::curlyhat(knot2_->varpi->getValue())*J_21_inv->getValue();

      // 6 x 6 Pose Jacobian 1
      if(!knot1_->T_k0->isLocked()) {

        // Add Jacobian
        Eigen::MatrixXd jacobian = -w * T_21->getValue().adjoint() + T_i1->getValue().adjoint();
        outJacobians->push_back(Jacobian(knot1_->T_k0->getKey(), lhs*jacobian));
      }

      // 6 x 6 Pose Jacobian 2
      if(!knot2_->T_k0->isLocked()) {

        // Add Jacobian
        outJacobians->push_back(Jacobian(knot2_->T_k0->getKey(), lhs*w));
      }
    }

    // 6 x 6 Velocity Jacobian 1
    if(!knot1_->varpi->isLocked()) {

      // Add Jacobian
      outJacobians->push_back(Jacobian(knot1_->varpi->getKey(), lhs*lambda12_*J_i1));
    }

    // 6 x 6 Velocity Jacobian 2
    if(!knot2_->varpi->isLocked()) {

      // Add Jacobian
      Eigen::MatrixXd jacobian = psi12_*J_i1*J_21_inv->getValue();
      outJacobians->push_back(Jacobian(knot2_->varpi->getKey(), lhs*jacobian));
    }
  }
}

} // se3
} // steam
