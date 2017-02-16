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
  Eigen::Matrix<double,3,3>  R_model_b_a = Euler2Rotation(model_rpy);

  // Eigen::Matrix<double,3,3> result_error;
  // result_error << R_meas_b_a*R_model_b_a;

  Eigen::Matrix<double,1,1> result_error;
  result_error << (0);

  return result_error;
}

// Jacobian is simply from pre-existing blockauto function for computing Jacobian of 4x4 transformation Matrix
// This will require that first we convert our 3x3 into a 4x4 (should this be done before error calc... probably)
// So to convert a 3x3 matrix to a 4x4, you simply copy in the values for the 3x3 upper left block, like so:
// [ a11  a12  a13 ]
// [ a21  a22  a23 ]
// [ a31  a32  a33 ]
// That 3x3 becomes this 4x4:
// [ a11  a12  a13  0 ]
// [ a21  a22  a23  0 ]
// [ a31  a32  a33  0 ]
// [   0    0    0  1 ]

} // steam
