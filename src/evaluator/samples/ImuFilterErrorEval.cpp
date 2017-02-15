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

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
ImuFilterErrorEval::ImuFilterErrorEval(const double & imu_meas_roll,
                                       const double & imu_meas_pitch,
                                       const double & imu_meas_yaw,
                                       const se3::TransformEvaluator::ConstPtr& T_b_a)
  : imu_meas_roll_(imu_meas_roll), imu_meas_pitch_(imu_meas_pitch), imu_meas_yaw_(imu_meas_yaw), T_b_a_(T_b_a) {
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

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,1> ImuFilterErrorEval::evaluate() const {
  auto T_b_a = T_b_a_->evaluate();
  auto component_tf = componentTF(T_b_a);
  // double meas_distance_sq = pow(tether_meas_a_,2) + pow(tether_meas_b_,2);
  // meas_distance_sq -= ( 2.0 * tether_meas_a_ * tether_meas_b_ * cos(component_tf.yaw) );
  // double meas_distance = sqrt(meas_distance_sq);

  Eigen::Matrix<double,1,1> result_error;
  result_error << (0);

  return result_error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Interface for the general 'evaluation', with Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,1> ImuFilterErrorEval::evaluate(const Eigen::Matrix<double,1,1>& lhs,
                        std::vector<Jacobian<1,6> >* jacs) const {

  // Check and initialize jacobian array
  if (jacs == NULL) {
    throw std::invalid_argument("Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  // Get evaluation tree
  auto blkAutoEvalT_b_a =T_b_a_->getBlockAutomaticEvaluation();

  // Get evaluation from the tree
  auto T_b_a = blkAutoEvalT_b_a.getValue();

  auto component_tf = componentTF(T_b_a);
  // double meas_distance_sq = pow(tether_meas_a_,2) + pow(tether_meas_b_,2);
  // meas_distance_sq -= ( 2.0 * tether_meas_a_ * tether_meas_b_ * cos(component_tf.yaw) );
  // double meas_distance = sqrt(meas_distance_sq);

  // Get Jacobians
  Eigen::Matrix<double,1,6> tether_err_jac = TetherModelJacobian(component_tf);
  auto newLhs = lhs*tether_err_jac;
  T_b_a_->appendBlockAutomaticJacobians(newLhs, blkAutoEvalT_b_a.getRoot(), jacs);

  Eigen::Matrix<double,1,1> result_error;
  result_error << (0);

  return result_error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Error Jacobian (includes measured distance since it contains part of the state)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,6> ImuFilterErrorEval::TetherModelJacobian(ComponentsTF &component_tf) const {

  // DEBUGGING: Print Statement
  std::cout << "\n";
  std::cout << "meas_roll:\t" << imu_meas_roll_ << "\n";
  std::cout << "meas_pitch:\t" << imu_meas_pitch_ << "\n";
  std::cout << "meas_yaw:\t" << imu_meas_yaw_ << "\n";
  std::cout << "model_roll:\t" << component_tf.roll << "\n";
  std::cout << "model_pitch:\t" << component_tf.pitch << "\n";
  std::cout << "model_yaw:\t" << component_tf.yaw << "\n";
  std::cout << "\n";

  // Construct Jacobian with respect to x, y, z, and rotation (yaw)
  Eigen::Matrix<double,1,6> jac;
  // jac << dx, dy, dz, droll, dpitch, dyaw;
  jac << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  return jac;
}

} // steam
