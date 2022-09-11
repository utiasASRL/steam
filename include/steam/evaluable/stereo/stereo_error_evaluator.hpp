#pragma once

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/stereo/compose_landmark_evaluator.hpp"

namespace steam {
namespace stereo {

/** \brief Simple structure to hold the stereo camera intrinsics */
struct CameraIntrinsics {
  using Ptr = std::shared_ptr<CameraIntrinsics>;
  using ConstPtr = std::shared_ptr<const CameraIntrinsics>;

  /** \brief Stereo baseline */
  double b;
  /** \brief Focal length in the u-coordinate (horizontal) */
  double fu;
  /** \brief Focal length in the v-coordinate (vertical) */
  double fv;
  /** \brief Focal center offset in the u-coordinate (horizontal) */
  double cu;
  /** \brief Focal center offset in the v-coordinate (vertical) */
  double cv;
};

/** \brief Stereo camera error function evaluator */
class StereoErrorEvaluator : public Evaluable<Eigen::Vector4d> {
 public:
  using Ptr = std::shared_ptr<StereoErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const StereoErrorEvaluator>;

  using PoseInType = lgmath::se3::Transformation;
  using LmInType = Eigen::Vector4d;
  using OutType = Eigen::Vector4d;

  static Ptr MakeShared(const Eigen::Vector4d& meas,
                        const CameraIntrinsics::ConstPtr& intrinsics,
                        const Evaluable<PoseInType>::ConstPtr& T_cam_landmark,
                        const Evaluable<LmInType>::ConstPtr& landmark);
  StereoErrorEvaluator(const Eigen::Vector4d& meas,
                       const CameraIntrinsics::ConstPtr& intrinsics,
                       const Evaluable<PoseInType>::ConstPtr& T_cam_landmark,
                       const Evaluable<LmInType>::ConstPtr& landmark);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  Eigen::Vector4d cameraModel(const Eigen::Vector4d& point) const;
  Eigen::Matrix4d cameraModelJacobian(const Eigen::Vector4d& point) const;

 private:
  /** \brief Measurement coordinates extracted from images (ul vl ur vr) */
  const Eigen::Vector4d meas_;

  /** \brief Camera instrinsics */
  const CameraIntrinsics::ConstPtr intrinsics_;

  /**
   * \brief Point evaluator
   * \details evaluates the point transformed into the camera frame
   */
  const ComposeLandmarkEvaluator::ConstPtr eval_;
};
}  // end namespace stereo
}  // namespace steam
