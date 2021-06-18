//////////////////////////////////////////////////////////////////////////////////////////////
/// \file ImuErrorEval.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/samples/ImuErrorEval.hpp>
namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ImuErrorEval::ImuErrorEval(
    const Eigen::Matrix<double,6,1>& measurement,
    const Eigen::Matrix3d& C_body_enu,
    const VectorSpaceStateVar::ConstPtr& varpi,
    const VectorSpaceStateVar::ConstPtr& varpi_dot,
    const VectorSpaceStateVar::ConstPtr& imu_bias,
    const lgmath::se3::Transformation& T_s_v)
  : meas_(measurement), C_body_enu_(C_body_enu), varpi_(varpi), varpi_dot_(varpi_dot), imu_bias_(imu_bias), adT_s_v_(T_s_v.adjoint()) {
    gravity_=Eigen::Vector3d::Zero();
    gravity_(2)=9.81;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool ImuErrorEval::isActive() const {
  return !varpi_->isLocked() || !varpi_dot_->isLocked() || !imu_bias_->isLocked();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> ImuErrorEval::evaluate() const {
  Eigen::Matrix<double,6,1> error;
  error.head<3>() = meas_.head<3>() + adT_s_v_.bottomRightCorner<3,3>()*varpi_->getValue().tail<3>() - imu_bias_->getValue().head<3>();
  error.tail<3>() = meas_.tail<3>() + adT_s_v_.topLeftCorner<3,3>()*varpi_dot_->getValue().head<3>() + adT_s_v_.topRightCorner<3,3>()*varpi_->getValue().tail<3>()
                    - imu_bias_->getValue().tail<3>() - C_body_enu_*gravity_;
  return error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error and relevant Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,6,1> ImuErrorEval::evaluate(
    const Eigen::Matrix<double,6,6>& lhs,
    std::vector<Jacobian<6,6> >* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Check that dimensions match
  // if (lhs.cols() != stateVec_->getPerturbDim()) {
  //   throw std::runtime_error("evaluate had dimension mismatch.");
  // }

  // Construct Jacobian
  if(!varpi_->isLocked()) {
    // Construct Jacobian Object
    jacs->push_back(Jacobian<6,6>());
    Jacobian<6,6>& jacref = jacs->back();
    jacref.key = varpi_->getKey();
    Eigen::MatrixXd jacobian(6,6);
    jacobian.setZero();
    jacobian.topRightCorner<3,3>()=adT_s_v_.bottomRightCorner<3,3>();//Eigen::Matrix3d::Identity();
    jacref.jac = lhs * jacobian;
  }

  if(!varpi_dot_->isLocked()) {
    // Construct Jacobian Object
    jacs->push_back(Jacobian<6,6>());
    Jacobian<6,6>& jacref = jacs->back();
    jacref.key = varpi_dot_->getKey();
    Eigen::MatrixXd jacobian(6,6);
    jacobian.setZero();
    jacobian.bottomLeftCorner<3,3>()=adT_s_v_.topLeftCorner<3,3>();//Eigen::Matrix3d::Identity();
    jacobian.bottomRightCorner<3,3>()=adT_s_v_.topRightCorner<3,3>();
    jacref.jac = lhs * jacobian;
  }

  if(!imu_bias_->isLocked()) {
    // Construct Jacobian Object
    jacs->push_back(Jacobian<6,6>());
    Jacobian<6,6>& jacref = jacs->back();
    jacref.key = imu_bias_->getKey();
    Eigen::MatrixXd jacobian(6,6);
    jacobian.setIdentity();
    jacref.jac = -lhs * jacobian;
  }


  // Return error
  Eigen::Matrix<double,6,1> error;
  error.head<3>() = meas_.head<3>() + adT_s_v_.bottomRightCorner<3,3>()*varpi_->getValue().tail<3>() - imu_bias_->getValue().head<3>();
  error.tail<3>() = meas_.tail<3>() + adT_s_v_.topLeftCorner<3,3>()*varpi_dot_->getValue().head<3>() + adT_s_v_.topRightCorner<3,3>()*varpi_->getValue().tail<3>()
                    - imu_bias_->getValue().tail<3>() - C_body_enu_*gravity_;
  return error;
}

} // steam
