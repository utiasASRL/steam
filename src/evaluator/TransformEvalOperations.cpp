//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvalOperations.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/TransformEvalOperations.hpp>

#include <lgmath/SE3.hpp>
#include <glog/logging.h>

namespace steam {
namespace se3 {

/// Compose

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeTransformEvaluator::ComposeTransformEvaluator(const TransformEvaluator::ConstPtr& pose1,
                                                     const TransformEvaluator::ConstPtr& pose2)
  : pose1_(pose1), pose2_(pose2) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeTransformEvaluator::Ptr ComposeTransformEvaluator::MakeShared(const TransformEvaluator::ConstPtr& pose1,
                                                                     const TransformEvaluator::ConstPtr& pose2) {
  return ComposeTransformEvaluator::Ptr(new ComposeTransformEvaluator(pose1, pose2));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool ComposeTransformEvaluator::isActive() const {
  return pose1_->isActive() || pose2_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant transformation matrix (pose1*pose2)
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation ComposeTransformEvaluator::evaluate() const {
  return pose1_->evaluate()*pose2_->evaluate();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant transformation matrix, and Jacobians w.r.t. state variables
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation ComposeTransformEvaluator::evaluate(std::vector<Jacobian>* jacs) const {

  // Evaluate
  std::vector<Jacobian> jacs1;
  lgmath::se3::Transformation pose1 = pose1_->evaluate(&jacs1);

  std::vector<Jacobian> jacs2;
  lgmath::se3::Transformation pose2 = pose2_->evaluate(&jacs2);

  // Check and initialize jacobian array
  CHECK_NOTNULL(jacs);
  jacs->clear();
  jacs->reserve(jacs1.size() + jacs2.size());

  // Jacobians 1
  jacs->insert(jacs->end(), jacs1.begin(), jacs1.end());
  unsigned int jacs1size = jacs->size();

  // Jacobians 2
  for (unsigned int j = 0; j < jacs2.size(); j++) {

    // Check if a jacobian w.r.t this state variable exists in 'other' half of transform evaluators
    // ** If a jacobian exists, we must add to it rather than creating a new entry
    unsigned int k = 0;
    for (; k < jacs1size; k++) {
      if ((*jacs)[k].key.equals(jacs2[j].key)) {
        break;
      }
    }

    // Add jacobian information
    if (k < jacs1size) {
      // Add to existing entry
      (*jacs)[k].jac += pose1.adjoint() * jacs2[j].jac;
    } else {
      // Create new entry
      jacs->push_back(Jacobian());
      size_t i = jacs->size() - 1;
      (*jacs)[i].key = jacs2[j].key;
      (*jacs)[i].jac = pose1.adjoint() * jacs2[j].jac;
    }
  }

  // Return composition
  return pose1*pose2;
}

/// Inverse

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
InverseTransformEvaluator::InverseTransformEvaluator(const TransformEvaluator::ConstPtr& pose) : pose_(pose) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
InverseTransformEvaluator::Ptr InverseTransformEvaluator::MakeShared(const TransformEvaluator::ConstPtr& pose) {
  return InverseTransformEvaluator::Ptr(new InverseTransformEvaluator(pose));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool InverseTransformEvaluator::isActive() const {
  return pose_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation InverseTransformEvaluator::evaluate() const {
  return pose_->evaluate().inverse();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant transformation matrix, and Jacobians w.r.t. state variables
//////////////////////////////////////////////////////////////////////////////////////////////
lgmath::se3::Transformation InverseTransformEvaluator::evaluate(std::vector<Jacobian>* jacs) const {

  // Evaluate
  std::vector<Jacobian> jacsCopy;
  lgmath::se3::Transformation poseInverse = pose_->evaluate(&jacsCopy).inverse();

  // Check and initialize jacobian array
  CHECK_NOTNULL(jacs);
  jacs->clear();
  jacs->resize(jacsCopy.size());

  // Jacobians
  for (unsigned int j = 0; j < jacsCopy.size(); j++) {
    (*jacs)[j].key = jacsCopy[j].key;
    (*jacs)[j].jac = -poseInverse.adjoint() * jacsCopy[j].jac;
  }

  return poseInverse;
}


/// Log map

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
LogMapEvaluator::LogMapEvaluator(const TransformEvaluator::ConstPtr& pose) : pose_(pose) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
LogMapEvaluator::Ptr LogMapEvaluator::MakeShared(const TransformEvaluator::ConstPtr& pose) {
  return LogMapEvaluator::Ptr(new LogMapEvaluator(pose));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool LogMapEvaluator::isActive() const {
  return pose_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the resultant 6x1 vector belonging to the se(3) algebra
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> LogMapEvaluator::evaluate() const {
  return pose_->evaluate().vec();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the 6x1 vector belonging to the se(3) algebra and relevant Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> LogMapEvaluator::evaluate(std::vector<Jacobian>* jacs) const {

  // Evaluate
  std::vector<Jacobian> jacsCopy;
  Eigen::Matrix<double,6,1> vec = pose_->evaluate(&jacsCopy).vec();

  // Check and initialize jacobian array
  CHECK_NOTNULL(jacs);
  jacs->clear();
  jacs->resize(jacsCopy.size());

  // Jacobians
  for (unsigned int j = 0; j < jacsCopy.size(); j++) {
    (*jacs)[j].key = jacsCopy[j].key;
    (*jacs)[j].jac = lgmath::se3::vec2jacinv(vec) * jacsCopy[j].jac;
  }

  return vec;
}

/// Landmark

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeLandmarkEvaluator::ComposeLandmarkEvaluator(const TransformEvaluator::ConstPtr& pose,
                                                   const se3::LandmarkStateVar::ConstPtr& landmark)
  : landmark_(landmark) {

  // Check if landmark has a reference frame and create pose evaluator
  if(landmark_->hasReferenceFrame()) {
    pose_ = ComposeTransformEvaluator::MakeShared(pose, InverseTransformEvaluator::MakeShared(landmark_->getReferenceFrame()));
  } else {
    pose_ = pose;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Pseudo constructor - return a shared pointer to a new instance
//////////////////////////////////////////////////////////////////////////////////////////////
ComposeLandmarkEvaluator::Ptr ComposeLandmarkEvaluator::MakeShared(const TransformEvaluator::ConstPtr& pose,
                                                                   const se3::LandmarkStateVar::ConstPtr& landmark) {
  return ComposeLandmarkEvaluator::Ptr(new ComposeLandmarkEvaluator(pose, landmark));
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool ComposeLandmarkEvaluator::isActive() const {
  return pose_->isActive() || !landmark_->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the point transformed by the transform evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d ComposeLandmarkEvaluator::evaluate() const {
  return pose_->evaluate()*landmark_->getValue();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the point transformed by the transform evaluator and relevant Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector4d ComposeLandmarkEvaluator::evaluate(std::vector<Jacobian>* jacs) const {

  // Evaluate
  std::vector<Jacobian> poseJacs;
  lgmath::se3::Transformation pose = pose_->evaluate(&poseJacs);

  Eigen::Vector4d point_in_c = pose * landmark_->getValue();

  // Check and initialize jacobian array
  CHECK_NOTNULL(jacs);
  jacs->clear();
  jacs->reserve(poseJacs.size() + 1);

  // 4 x 6 Pose Jacobians
  for (unsigned int j = 0; j < poseJacs.size(); j++) {
    jacs->push_back(Jacobian());
    size_t i = jacs->size() - 1;
    (*jacs)[i].key = poseJacs[j].key;
    (*jacs)[i].jac = lgmath::se3::point2fs(point_in_c.head<3>()) * poseJacs[j].jac;
  }

  // 4 x 3 Landmark Jacobian
  if(!landmark_->isLocked()) {
    jacs->push_back(Jacobian());
    size_t i = jacs->size() - 1;
    (*jacs)[i].key = landmark_->getKey();
    (*jacs)[i].jac = pose.matrix() * Eigen::Matrix<double,4,3>::Identity();
  }

  // Return error
  return point_in_c;
}

} // se3
} // steam
