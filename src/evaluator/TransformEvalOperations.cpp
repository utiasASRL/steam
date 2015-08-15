//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvalOperations.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/TransformEvalOperations.hpp>

#include <lgmath.hpp>

namespace steam {
namespace se3 {

/// Compose

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeTransformEvaluator::ComposeTransformEvaluator(const TransformEvaluator::ConstPtr& transform1,
                                                     const TransformEvaluator::ConstPtr& transform2)
  : transform1_(transform1), transform2_(transform2) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeTransformEvaluator::Ptr ComposeTransformEvaluator::MakeShared(const TransformEvaluator::ConstPtr& transform1,
                                                                     const TransformEvaluator::ConstPtr& transform2) {
  return ComposeTransformEvaluator::Ptr(new ComposeTransformEvaluator(transform1, transform2));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool ComposeTransformEvaluator::isActive() const {
  return transform1_->isActive() || transform2_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant transformation matrix (transform1*transform2)
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation ComposeTransformEvaluator::evaluate() const {
  return transform1_->evaluate()*transform2_->evaluate();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* ComposeTransformEvaluator::evaluateTree() const {

  // Evaluate sub-trees
  EvalTreeNode<lgmath::se3::Transformation>* transform1 = transform1_->evaluateTree();
  EvalTreeNode<lgmath::se3::Transformation>* transform2 = transform2_->evaluateTree();

  // Make new root node -- note we get memory from the pool
  EvalTreeNode<lgmath::se3::Transformation>* root = EvalTreeNode<lgmath::se3::Transformation>::pool.getObj();
  root->setValue(transform1->getValue()*transform2->getValue());

  // Add children
  root->addChild(transform1);
  root->addChild(transform2);

  // Return new root node
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void ComposeTransformEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                  EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                  std::vector<Jacobian<> >* outJacobians) const {

  // Cast back to transformation
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform1_->isActive()) {
    transform1_->appendJacobians(lhs, t1, outJacobians);
  }

  // Get index of split between left and right-hand-side of Jacobians
  unsigned int hintIndex = outJacobians->size();

  // Check if transform2 is active
  if (transform2_->isActive()) {

    EvalTreeNode<lgmath::se3::Transformation>* t2 =
        static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(1));

    transform2_->appendJacobians(lhs*t1->getValue().adjoint(), t2, outJacobians);
  }

  // Merge jacobians
  Jacobian<>::merge(outJacobians, hintIndex);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void ComposeTransformEvaluator::appendJacobians1(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  // Cast back to transformation
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform1_->isActive()) {
    transform1_->appendJacobians1(lhs, t1, outJacobians);
  }

  // Get index of split between left and right-hand-side of Jacobians
  unsigned int hintIndex = outJacobians->size();

  // Check if transform2 is active
  if (transform2_->isActive()) {

    EvalTreeNode<lgmath::se3::Transformation>* t2 =
        static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(1));

    transform2_->appendJacobians1(lhs*t1->getValue().adjoint(), t2, outJacobians);
  }

  // Merge jacobians
  Jacobian<1,6>::merge(outJacobians, hintIndex);
}

void ComposeTransformEvaluator::appendJacobians2(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  // Cast back to transformation
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform1_->isActive()) {
    transform1_->appendJacobians2(lhs, t1, outJacobians);
  }

  // Get index of split between left and right-hand-side of Jacobians
  unsigned int hintIndex = outJacobians->size();

  // Check if transform2 is active
  if (transform2_->isActive()) {

    EvalTreeNode<lgmath::se3::Transformation>* t2 =
        static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(1));

    transform2_->appendJacobians2(lhs*t1->getValue().adjoint(), t2, outJacobians);
  }

  // Merge jacobians
  Jacobian<2,6>::merge(outJacobians, hintIndex);
}

void ComposeTransformEvaluator::appendJacobians3(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  // Cast back to transformation
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform1_->isActive()) {
    transform1_->appendJacobians3(lhs, t1, outJacobians);
  }

  // Get index of split between left and right-hand-side of Jacobians
  unsigned int hintIndex = outJacobians->size();

  // Check if transform2 is active
  if (transform2_->isActive()) {

    EvalTreeNode<lgmath::se3::Transformation>* t2 =
        static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(1));

    transform2_->appendJacobians3(lhs*t1->getValue().adjoint(), t2, outJacobians);
  }

  // Merge jacobians
  Jacobian<3,6>::merge(outJacobians, hintIndex);
}

void ComposeTransformEvaluator::appendJacobians4(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  // Cast back to transformation
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform1_->isActive()) {
    transform1_->appendJacobians4(lhs, t1, outJacobians);
  }

  // Get index of split between left and right-hand-side of Jacobians
  unsigned int hintIndex = outJacobians->size();

  // Check if transform2 is active
  if (transform2_->isActive()) {

    EvalTreeNode<lgmath::se3::Transformation>* t2 =
        static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(1));

    Eigen::Matrix<double,4,6> newLhs = lhs*t1->getValue().adjoint();
    transform2_->appendJacobians4(newLhs, t2, outJacobians);
  }

  // Merge jacobians
  Jacobian<4,6>::merge(outJacobians, hintIndex);
}

void ComposeTransformEvaluator::appendJacobians6(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  // Cast back to transformation
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform1_->isActive()) {
    transform1_->appendJacobians6(lhs, t1, outJacobians);
  }

  // Get index of split between left and right-hand-side of Jacobians
  unsigned int hintIndex = outJacobians->size();

  // Check if transform2 is active
  if (transform2_->isActive()) {

    EvalTreeNode<lgmath::se3::Transformation>* t2 =
        static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(1));

    transform2_->appendJacobians6(lhs*t1->getValue().adjoint(), t2, outJacobians);
  }

  // Merge jacobians
  Jacobian<6,6>::merge(outJacobians, hintIndex);
}

/// Inverse

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
InverseTransformEvaluator::InverseTransformEvaluator(const TransformEvaluator::ConstPtr& transform) : transform_(transform) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
InverseTransformEvaluator::Ptr InverseTransformEvaluator::MakeShared(const TransformEvaluator::ConstPtr& transform) {
  return InverseTransformEvaluator::Ptr(new InverseTransformEvaluator(transform));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool InverseTransformEvaluator::isActive() const {
  return transform_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation InverseTransformEvaluator::evaluate() const {
  return transform_->evaluate().inverse();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the transformation matrix tree
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<lgmath::se3::Transformation>* InverseTransformEvaluator::evaluateTree() const {

  // Evaluate sub-trees
  EvalTreeNode<lgmath::se3::Transformation>* transform = transform_->evaluateTree();

  // Make new root node -- note we get memory from the pool
  EvalTreeNode<lgmath::se3::Transformation>* root = EvalTreeNode<lgmath::se3::Transformation>::pool.getObj();
  root->setValue(transform->getValue().inverse());

  // Add children
  root->addChild(transform);

  // Return new root node
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void InverseTransformEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                  EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                  std::vector<Jacobian<> >* outJacobians) const {

  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians((-1)*lhs*evaluationTree->getValue().adjoint(),
                                static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                outJacobians);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void InverseTransformEvaluator::appendJacobians1(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians1((-1)*lhs*evaluationTree->getValue().adjoint(),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void InverseTransformEvaluator::appendJacobians2(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians2((-1)*lhs*evaluationTree->getValue().adjoint(),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void InverseTransformEvaluator::appendJacobians3(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians3((-1)*lhs*evaluationTree->getValue().adjoint(),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void InverseTransformEvaluator::appendJacobians4(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    Eigen::Matrix<double,4,6> newLhs = (-1)*lhs*evaluationTree->getValue().adjoint();
    transform_->appendJacobians4(newLhs,
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void InverseTransformEvaluator::appendJacobians6(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians6((-1)*lhs*evaluationTree->getValue().adjoint(),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

/// Log map

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
LogMapEvaluator::LogMapEvaluator(const TransformEvaluator::ConstPtr& transform) : transform_(transform) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
LogMapEvaluator::Ptr LogMapEvaluator::MakeShared(const TransformEvaluator::ConstPtr& transform) {
  return LogMapEvaluator::Ptr(new LogMapEvaluator(transform));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool LogMapEvaluator::isActive() const {
  return transform_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant 6x1 vector belonging to the se(3) algebra
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> LogMapEvaluator::evaluate() const {
  return transform_->evaluate().vec();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant 6x1 vector belonging to the se(3) algebra and
///        sub-tree of evaluations
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<Eigen::Matrix<double,6,1> >* LogMapEvaluator::evaluateTree() const {

  // Evaluate sub-trees
  EvalTreeNode<lgmath::se3::Transformation>* transform = transform_->evaluateTree();

  // Make new root node -- note we get memory from the pool
  EvalTreeNode<Eigen::Matrix<double,6,1> >* root = EvalTreeNode<Eigen::Matrix<double,6,1> >::pool.getObj();
  root->setValue(transform->getValue().vec());

  // Add children
  root->addChild(transform);

  // Return new root node
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void LogMapEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                  EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                                  std::vector<Jacobian<> >* outJacobians) const {

  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians(lhs * lgmath::se3::vec2jacinv(evaluationTree->getValue()),
                                static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                outJacobians);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void LogMapEvaluator::appendJacobians1(const Eigen::Matrix<double,1,6>& lhs,
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians1(lhs * lgmath::se3::vec2jacinv(evaluationTree->getValue()),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void LogMapEvaluator::appendJacobians2(const Eigen::Matrix<double,2,6>& lhs,
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians2(lhs * lgmath::se3::vec2jacinv(evaluationTree->getValue()),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void LogMapEvaluator::appendJacobians3(const Eigen::Matrix<double,3,6>& lhs,
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians3(lhs * lgmath::se3::vec2jacinv(evaluationTree->getValue()),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void LogMapEvaluator::appendJacobians4(const Eigen::Matrix<double,4,6>& lhs,
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians4(lhs * lgmath::se3::vec2jacinv(evaluationTree->getValue()),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

void LogMapEvaluator::appendJacobians6(const Eigen::Matrix<double,6,6>& lhs,
                              EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  // Check if transform is active
  if (transform_->isActive()) {
    transform_->appendJacobians6(lhs * lgmath::se3::vec2jacinv(evaluationTree->getValue()),
                                 static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0)),
                                 outJacobians);
  }
}

/// Landmark

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeLandmarkEvaluator::ComposeLandmarkEvaluator(const TransformEvaluator::ConstPtr& transform,
                                                   const se3::LandmarkStateVar::ConstPtr& landmark)
  : landmark_(landmark) {

  // Check if landmark has a reference frame and create pose evaluator
  if(landmark_->hasReferenceFrame()) {
    transform_ = ComposeTransformEvaluator::MakeShared(transform, InverseTransformEvaluator::MakeShared(landmark_->getReferenceFrame()));
  } else {
    transform_ = transform;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeLandmarkEvaluator::Ptr ComposeLandmarkEvaluator::MakeShared(const TransformEvaluator::ConstPtr& transform,
                                                                   const se3::LandmarkStateVar::ConstPtr& landmark) {
  return ComposeLandmarkEvaluator::Ptr(new ComposeLandmarkEvaluator(transform, landmark));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool ComposeLandmarkEvaluator::isActive() const {
  return transform_->isActive() || !landmark_->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the point transformed by the transform evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d ComposeLandmarkEvaluator::evaluate() const {
  return transform_->evaluate()*landmark_->getValue();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the point transformed by the transform evaluator and
///        sub-tree of evaluations
//////////////////////////////////////////////////////////////////////////////////////////////
EvalTreeNode<Eigen::Vector4d>* ComposeLandmarkEvaluator::evaluateTree() const {

  // Evaluate transform sub-tree
  EvalTreeNode<lgmath::se3::Transformation>* transform = transform_->evaluateTree();

  // Make new leaf node for landmark state variable -- note we get memory from the pool
  EvalTreeNode<Eigen::Vector4d>* landmarkLeaf = EvalTreeNode<Eigen::Vector4d>::pool.getObj();
  landmarkLeaf->setValue(landmark_->getValue());

  // Make new root node -- note we get memory from the pool
  EvalTreeNode<Eigen::Vector4d>* root = EvalTreeNode<Eigen::Vector4d>::pool.getObj();
  root->setValue(transform->getValue()*landmarkLeaf->getValue());

  // Add children
  root->addChild(transform);
  root->addChild(landmarkLeaf);

  // Return new root node
  return root;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void ComposeLandmarkEvaluator::appendJacobians(const Eigen::MatrixXd& lhs,
                                  EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                                  std::vector<Jacobian<> >* outJacobians) const {

  // Cast back to transform
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform_->isActive()) {
    const Eigen::Vector4d& homogeneous = evaluationTree->getValue();
    transform_->appendJacobians(lhs * lgmath::se3::point2fs(homogeneous.head<3>(), homogeneous[3]),
                                t1, outJacobians);
  }

  // Check if state is locked
  if (!landmark_->isLocked()) {

    // Check that dimensions match
    if (lhs.cols() != 4) {
      throw std::runtime_error("appendJacobians had dimension mismatch.");
    }

    // Add Jacobian -- transform.matrix() * Eigen::Matrix<double,4,3>::Identity()
    outJacobians->push_back(Jacobian<>(landmark_->getKey(), lhs * t1->getValue().matrix().block<4,3>(0,0)));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Fixed-size evaluations of the Jacobian tree
//////////////////////////////////////////////////////////////////////////////////////////////
void ComposeLandmarkEvaluator::appendJacobians1(const Eigen::Matrix<double,1,4>& lhs,
                              EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                              std::vector<Jacobian<1,6> >* outJacobians) const {
  // Cast back to transform
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform_->isActive()) {
    const Eigen::Vector4d& homogeneous = evaluationTree->getValue();
    Eigen::Matrix<double,1,6> newLhs = lhs * lgmath::se3::point2fs(homogeneous.head<3>(), homogeneous[3]);
    transform_->appendJacobians1(newLhs, t1, outJacobians);
  }

  // Check if state is locked
  if (!landmark_->isLocked()) {

    // Add Jacobian -- transform.matrix() * Eigen::Matrix<double,4,3>::Identity()
    outJacobians->push_back(Jacobian<1,6>());
    Jacobian<1,6>& ref = outJacobians->back();
    ref.key = landmark_->getKey();
    ref.jac.block<1,3>(0,0) = lhs * t1->getValue().matrix().block<4,3>(0,0);
  }
}

void ComposeLandmarkEvaluator::appendJacobians2(const Eigen::Matrix<double,2,4>& lhs,
                              EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                              std::vector<Jacobian<2,6> >* outJacobians) const {
  // Cast back to transform
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform_->isActive()) {
    const Eigen::Vector4d& homogeneous = evaluationTree->getValue();
    Eigen::Matrix<double,2,6> newLhs = lhs * lgmath::se3::point2fs(homogeneous.head<3>(), homogeneous[3]);
    transform_->appendJacobians2(newLhs, t1, outJacobians);
  }

  // Check if state is locked
  if (!landmark_->isLocked()) {

    // Add Jacobian -- transform.matrix() * Eigen::Matrix<double,4,3>::Identity()
    outJacobians->push_back(Jacobian<2,6>());
    Jacobian<2,6>& ref = outJacobians->back();
    ref.key = landmark_->getKey();
    ref.jac.block<2,3>(0,0) = lhs * t1->getValue().matrix().block<4,3>(0,0);
  }
}

void ComposeLandmarkEvaluator::appendJacobians3(const Eigen::Matrix<double,3,4>& lhs,
                              EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                              std::vector<Jacobian<3,6> >* outJacobians) const {
  // Cast back to transform
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform_->isActive()) {
    const Eigen::Vector4d& homogeneous = evaluationTree->getValue();
    Eigen::Matrix<double,3,6> newLhs = lhs * lgmath::se3::point2fs(homogeneous.head<3>(), homogeneous[3]);
    transform_->appendJacobians3(newLhs, t1, outJacobians);
  }

  // Check if state is locked
  if (!landmark_->isLocked()) {

    // Add Jacobian -- transform.matrix() * Eigen::Matrix<double,4,3>::Identity()
    outJacobians->push_back(Jacobian<3,6>());
    Jacobian<3,6>& ref = outJacobians->back();
    ref.key = landmark_->getKey();
    ref.jac.block<3,3>(0,0) = lhs * t1->getValue().matrix().block<4,3>(0,0);
  }
}

void ComposeLandmarkEvaluator::appendJacobians4(const Eigen::Matrix<double,4,4>& lhs,
                              EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                              std::vector<Jacobian<4,6> >* outJacobians) const {
  // Cast back to transform
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform_->isActive()) {
    const Eigen::Vector4d& homogeneous = evaluationTree->getValue();
    Eigen::Matrix<double,4,6> newLhs = lhs * lgmath::se3::point2fs(homogeneous.head<3>(), homogeneous[3]);
    transform_->appendJacobians4(newLhs, t1, outJacobians);
  }

  // Check if state is locked
  if (!landmark_->isLocked()) {

    // Add Jacobian -- transform.matrix() * Eigen::Matrix<double,4,3>::Identity()
    outJacobians->push_back(Jacobian<4,6>());
    Jacobian<4,6>& ref = outJacobians->back();
    ref.key = landmark_->getKey();
    ref.jac.block<4,3>(0,0) = lhs * t1->getValue().matrix().block<4,3>(0,0);
  }
}

void ComposeLandmarkEvaluator::appendJacobians6(const Eigen::Matrix<double,6,4>& lhs,
                              EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                              std::vector<Jacobian<6,6> >* outJacobians) const {
  // Cast back to transform
  EvalTreeNode<lgmath::se3::Transformation>* t1 =
      static_cast<EvalTreeNode<lgmath::se3::Transformation>*>(evaluationTree->childAt(0));

  // Check if transform1 is active
  if (transform_->isActive()) {
    const Eigen::Vector4d& homogeneous = evaluationTree->getValue();
    Eigen::Matrix<double,6,6> newLhs = lhs * lgmath::se3::point2fs(homogeneous.head<3>(), homogeneous[3]);
    transform_->appendJacobians6(newLhs, t1, outJacobians);
  }

  // Check if state is locked
  if (!landmark_->isLocked()) {

    // Add Jacobian -- transform.matrix() * Eigen::Matrix<double,4,3>::Identity()
    outJacobians->push_back(Jacobian<6,6>());
    Jacobian<6,6>& ref = outJacobians->back();
    ref.key = landmark_->getKey();
    ref.jac.block<6,3>(0,0) = lhs * t1->getValue().matrix().block<4,3>(0,0);
  }
}

} // se3
} // steam
