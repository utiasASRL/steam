//////////////////////////////////////////////////////////////////////////////////////////////
/// \file ImuFilterErrorEval.cpp
///
/// \author Patrick McGarey, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

//TODO: get correct size for Eigen matrices below, create error function, use TF Jacobian eval already in STEAM

#include <steam/evaluator/samples/ImuFilterErrorEval.hpp>
//added for debugging only
#include <iostream>

namespace steam {

//Declare Variables for Rotation Matrix
Eigen::Matrix<double,3,3> R_x;
Eigen::Matrix<double,3,3> R_y;
Eigen::Matrix<double,3,3> R_z;
Eigen::Matrix<double,3,3> R;

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ImuFilterErrorEval::ImuFilterErrorEval(const std::vector<double> & imu_meas_rpy_a,
                                       const std::vector<double> & imu_meas_rpy_b,
                                       const se3::TransformEvaluator::ConstPtr & T_b_a)
  : imu_meas_rpy_a_(imu_meas_rpy_a), imu_meas_rpy_b_(imu_meas_rpy_b), T_b_a_(T_b_a) {
}

ImuFilterErrorEval::~ImuFilterErrorEval() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables.
//////////////////////////////////////////////////////////////////////////////////////////////
bool ImuFilterErrorEval::isActive() const {
  return T_b_a_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Decomposes the TF into components.
//////////////////////////////////////////////////////////////////////////////////////////////
ComponentsTF ImuFilterErrorEval::componentTF(lgmath::se3::Transformation &tf) const {
  // TODO: Check out other ways to compute displacement and yaw.
  // example: tf.r_ba_ina()
  ComponentsTF component_tf;
  auto se3Vec = tf.vec();
  component_tf.roll =  se3Vec(3);
  component_tf.pitch = se3Vec(4);
  component_tf.yaw =   se3Vec(5);
  return component_tf;
}

// Calculates rotation matrix given euler angles.

Eigen::Matrix<double,3,3> Euler2Rotation(std::vector<double> rpy)
{
    // Calculate rotation about x axis
    R_x << 1, 0,           0,
           0, cos(rpy[0]), -sin(rpy[0]),
           0, sin(rpy[0]),  cos(rpy[0]);
    // Calculate rotation about y axis
    R_y << cos(rpy[1]),  0, sin(rpy[1]),
           0,            1, 0,
          -sin(rpy[1]),  0, cos(rpy[1]);

    // Calculate rotation about z axis
    R_z << cos(rpy[2]), -sin(rpy[2]), 0,
           sin(rpy[2]),  cos(rpy[2]), 0,
           0,           0,            1;
    // Combined rotation matrix
    R = R_z * R_y * R_x;

    return R;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,1> ImuFilterErrorEval::evaluate() const {
  auto T_b_a = T_b_a_->evaluate();
  auto component_tf = componentTF(T_b_a);

  // Get Rotation Matrix for two IMU Filter states
  Eigen::Matrix<double,3,3> R_meas_a = Euler2Rotation(imu_meas_rpy_a_);
  Eigen::Matrix<double,3,3> R_meas_b = Euler2Rotation(imu_meas_rpy_b_);

  // Calculate a Relative rotation from b to a
  Eigen::Matrix<double,3,3> R_meas_a_inv = R_meas_a.inverse();
  Eigen::Matrix<double,3,3> R_meas_b_a = R_meas_b*R_meas_a_inv;

  // Get Rotation Matrix for Model (function accepts vector of doubles)
  std::vector<double> model_rpy = {component_tf.roll,component_tf.pitch,component_tf.yaw};
  Eigen::Matrix<double,3,3> R_model_b_a = Euler2Rotation(model_rpy);

  // Get Rotation Error
  Eigen::Matrix<double,3,3> R_error = R_meas_b_a*R_model_b_a.inverse();

  // convert result to a 4x4 transformation matrix to use with blockauto Jacobian
  Eigen::Matrix<double,4,4> T_error;
  // Set to Identity to make bottom row of Matrix 0,0,0,1
  T_error.setIdentity();
  // set rotation matrix as upper 3x3 in a 4x4
  T_error.block<3,3>(0,0) = R_error;

  Eigen::Matrix<double,1,1> reutrn_error;
  reutrn_error << (0);

  return reutrn_error;
}

// Jacobian is simply from pre-existing blockauto function for computing Jacobian of 4x4 transformation Matrix
// http://stackoverflow.com/questions/25504397/eigen-combine-rotation-and-translation-into-one-matrix

} // steam
