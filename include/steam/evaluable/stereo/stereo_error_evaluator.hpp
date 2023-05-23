#pragma once
#include <iostream>

#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/stereo/compose_landmark_evaluator.hpp"
#include "steam/evaluable/se3/se3_state_var.hpp"

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


Eigen::Vector4d cameraModel(const CameraIntrinsics::ConstPtr& intrinsics, const Eigen::Vector4d& point);
Eigen::Matrix4d cameraModelJacobian(const CameraIntrinsics::ConstPtr& intrinsics, const Eigen::Vector4d& point);

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

/** \brief Evaluates the noise of an uncertain map landmark, which has been reprojected into the
        a query coordinate frame into the camera pixels. */
class LandmarkNoiseEvaluator : public Evaluable<Eigen::Matrix<double, 4, 4>> {
 public:

  /// \brief Constructor
  /// \param The landmark mean, in the query frame.
  /// \param The landmark covariance, in the query frame.
  /// \param The noise on the landmark measurement.
  /// \param The stereo camera intrinsics.
  /// \param The steam transform evaluator that takes points from the landmark frame
  ///        into the query frame.
  LandmarkNoiseEvaluator(const Eigen::Vector4d& landmark_mean,
                         const Eigen::Matrix3d& landmark_cov,
                         const Eigen::Matrix4d& meas_noise,
                         const CameraIntrinsics::ConstPtr& intrinsics,
                         const typename Evaluable<lgmath::se3::Transformation>::ConstPtr& T_query_map);

  ~LandmarkNoiseEvaluator()=default;
  
  /// \brief Evaluates the reprojection covariance 
  /// @return the 4x4 covariance of the landmark reprojected into the query stereo
  ///         camera frame.
  Eigen::Matrix4d value() const;

  bool active() const;

  void getRelatedVarKeys(KeySet& keys) const;
  Node<Eigen::Matrix4d>::Ptr forward() const;
  void backward(const Eigen::MatrixXd& lhs,
                        const Node<Eigen::Matrix4d>::Ptr& node,
                        Jacobians& jacs) const;

 private:
  template <int N>
  bool positiveDefinite(const Eigen::Matrix<double,N,N> &matrix) const {
  
    // Initialize an eigen value solver
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,N,N>> 
       eigsolver(matrix, Eigen::EigenvaluesOnly);

    // Check the minimum eigen value
    auto positive_definite = eigsolver.eigenvalues().minCoeff() > 0;
    if (!positive_definite) {
      std::cout << "Covariance \n" << matrix << "\n must be positive definite. "
                << "Min. eigenvalue : " << eigsolver.eigenvalues().minCoeff();
    }
    
    return positive_definite;
  }

  /// \brief The stereo camera intrinsics.
  CameraIntrinsics::ConstPtr intrinsics_;

  /// \brief The landmark covariance.
  Eigen::Matrix4d meas_noise_;

  /// \brief the landmark mean.
  Eigen::Vector4d mean_;

  /// \brief The steam transform evaluator that takes points from the landmark frame
  ///        into the query frame.
  Evaluable<lgmath::se3::Transformation>::ConstPtr T_query_map_;

  /// \brief The 3x3 landmark covariance (phi) dialated into a 3x3 matrix.
  /// @details dialated_phi_ = D*phi*D^T, where D is a 4x3 dialation matrix.
  Eigen::Matrix4d dialated_phi_;
};



}  // end namespace stereo
}  // namespace steam
