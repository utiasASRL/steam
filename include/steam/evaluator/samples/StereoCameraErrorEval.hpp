//////////////////////////////////////////////////////////////////////////////////////////////
/// \file StereoCameraErrorEval.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_STEREO_CAMERA_ERROR_EVALUATOR_HPP
#define STEAM_STEREO_CAMERA_ERROR_EVALUATOR_HPP

#include <steam.hpp>
#include <steam/problem/NoiseModel.hpp>

namespace steam {

namespace stereo {
  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Simple structure to hold the basic camera intrinsics
  //////////////////////////////////////////////////////////////////////////////////////////////
  struct CameraIntrinsics {

    /// Convenience typedefs
    typedef boost::shared_ptr<CameraIntrinsics> Ptr;
    typedef boost::shared_ptr<const CameraIntrinsics> ConstPtr;

    /// \brief Stereo baseline
    double b;

    /// \brief Focal length in the u-coordinate (horizontal)
    double fu;

    /// \brief Focal length in the v-coordinate (vertical)
    double fv;

    /// \brief Focal center offset in the u-coordinate (horizontal)
    double cu;

    /// \brief Focal center offset in the v-coordinate (vertical)
    double cv;
  };
}

class StereoLandmarkNoiseEvaluator : public NoiseEvaluator<4> {
  public:
    StereoLandmarkNoiseEvaluator(const Eigen::Vector4d& landmark_mean,
                                 const Eigen::Matrix3d& landmark_cov,
                                 const Eigen::Matrix4d& meas_noise,
                                 const stereo::CameraIntrinsics::ConstPtr& intrinsics,
                                 const se3::TransformEvaluator::ConstPtr& T_cam_landmark) {
      intrinsics_ = intrinsics;
      dialated_phi_.setZero();
      dialated_phi_.block(0,0,3,3) = landmark_cov;
      meas_noise_ = meas_noise;
      mean_ = landmark_mean;
      T_cam_landmark_ = T_cam_landmark;
     // LOG(INFO) << "landmark mean\n" << landmark_mean.hnormalized() << "\n";
    }
    ~StereoLandmarkNoiseEvaluator()=default;
  
  virtual Eigen::Matrix<double,4,4> evaluateCovariance() {
    // TODO: Check to see if we need to recaulculate;
    bool recalculate = true;
    if(recalculate) {
      const auto &T_l_p = T_cam_landmark_->evaluate();
      // 2. Calculate G
      
      camera_jacobian_j_ = cameraModelJacobian(T_l_p*mean_);
      auto lm_noise = camera_jacobian_j_ * T_l_p.matrix() * dialated_phi_ * 
                       T_l_p.matrix().transpose() * camera_jacobian_j_.transpose();
     // LOG(INFO) << "DPhiDt\n" << dialated_phi_ << "\n";
    //  LOG(INFO) << "Measurement noise: \n" << meas_noise_ << "\nLandmark noise:\n" << lm_noise << "\n";
      last_computed_cov_ = meas_noise_ + lm_noise;

    }
    return last_computed_cov_;
  }
  private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera model Jacobian
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d cameraModelJacobian(const Eigen::Vector4d& point) const {

    // Precompute values
    const double x = point[0];
    const double y = point[1];
    const double z = point[2];
    const double w = point[3];
    const double xr = x - w * intrinsics_->b;
    const double one_over_z = 1.0/z;
    const double one_over_z2 = one_over_z*one_over_z;

    // Construct Jacobian with respect to x, y, z, and scalar w
    const double dw = -intrinsics_->fu * intrinsics_->b * one_over_z;
    Eigen::Matrix4d jac;
    jac << intrinsics_->fu * one_over_z, 0.0, -intrinsics_->fu * x  * one_over_z2, 0.0,
           0.0, intrinsics_->fv * one_over_z, -intrinsics_->fv * y  * one_over_z2, 0.0,
           intrinsics_->fu * one_over_z, 0.0, -intrinsics_->fu * xr * one_over_z2,  dw,
           0.0, intrinsics_->fv * one_over_z, -intrinsics_->fv * y  * one_over_z2, 0.0;
    return jac;
  }

    se3::LandmarkStateVar::Ptr landmark_;
    stereo::CameraIntrinsics::ConstPtr intrinsics_;
    se3::TransformEvaluator::ConstPtr T_cam_landmark_;

    Eigen::Matrix4d dialated_phi_;
    Eigen::Matrix4d camera_jacobian_j_;
    Eigen::Vector4d mean_;
    Eigen::Matrix4d meas_noise_;

    Eigen::Matrix<double,4,4> last_computed_cov_;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Stereo camera error function evaluator
///
/// *Note that we fix MAX_STATE_DIM to 6. Typically the performance benefits of fixed size
///  matrices begin to die if larger than 6x6. Size 6 allows for transformation matrices
///  and 6D velocities. If you have a state-type larger than this, consider writing an
///  error evaluator that extends from the dynamically sized ErrorEvaluatorX.
//////////////////////////////////////////////////////////////////////////////////////////////
class StereoCameraErrorEval : public ErrorEvaluator<4,6>::type
{
public:

  /// Convenience typedefs
  typedef boost::shared_ptr<StereoCameraErrorEval> Ptr;
  typedef boost::shared_ptr<const StereoCameraErrorEval> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  StereoCameraErrorEval(const Eigen::Vector4d& meas,
                        const stereo::CameraIntrinsics::ConstPtr& intrinsics,
                        const se3::TransformEvaluator::ConstPtr& T_cam_landmark,
                        const se3::LandmarkStateVar::Ptr& landmark);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 4-d measurement error (ul vl ur vr)
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Vector4d evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 4-d measurement error (ul vl ur vr) and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Vector4d evaluate(const Eigen::Matrix4d& lhs,
                                   std::vector<Jacobian<4,6> >* jacs) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera model Jacobian
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d cameraModelJacobian() const { return cameraModelJacobian(meas_); };
private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera model
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Vector4d cameraModel(const Eigen::Vector4d& point) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera model Jacobian
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d cameraModelJacobian(const Eigen::Vector4d& point) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Measurement coordinates extracted from images (ul vl ur vr)
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Vector4d meas_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Camera instrinsics
  //////////////////////////////////////////////////////////////////////////////////////////////
  stereo::CameraIntrinsics::ConstPtr intrinsics_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Point evaluator (evaluates the point transformed into the camera frame)
  //////////////////////////////////////////////////////////////////////////////////////////////
  se3::ComposeLandmarkEvaluator::ConstPtr eval_;

};

} // steam

#endif // STEAM_STEREO_CAMERA_ERROR_EVALUATOR_HPP
