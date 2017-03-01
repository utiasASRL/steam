//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TetherLengthErrorEval.cpp
///
/// \author Patrick McGarey, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/samples/TetherLengthErrorEval.hpp>
//added for debugging only
#include <iostream>

namespace steam {

// intitialize doubles for Jacobian
double dx;
double dy;
double dz;
double dr;

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
TetherLengthErrorEval::TetherLengthErrorEval(const double & tether_length_a,
                                             const double & tether_length_b,
                                             const se3::TransformEvaluator::ConstPtr& T_b_a)
  : tether_meas_a_(tether_length_a), tether_meas_b_(tether_length_b), T_b_a_(T_b_a) {
}

TetherLengthErrorEval::~TetherLengthErrorEval() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns whether or not an evaluator contains unlocked state variables.
//////////////////////////////////////////////////////////////////////////////////////////////
bool TetherLengthErrorEval::isActive() const {
  return T_b_a_->isActive();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Decomposes the TF into components.
//////////////////////////////////////////////////////////////////////////////////////////////
DecomposedTF TetherLengthErrorEval::decomposeTF(lgmath::se3::Transformation &tf) const {
  DecomposedTF decomposed_tf;
  auto T = tf.C_ba();
  Eigen::Matrix3d R = T.matrix();
  double yaw;
  // check if singular
  double singular = sqrt(R(0,0) * R(0,0) +  R(1,0) * R(1,0));
  if (singular < 1e-6) {
    yaw = 0;
    std::cout << "WARN: singularity, setting yaw = 0" << "\n";
  }
  else {
    yaw = atan2(R(1,0),R(0,0));
  }
  decomposed_tf.yaw =  yaw;
  auto se3Vec = tf.vec();
  // decomposed_tf.yaw =  se3Vec(5); // this is not the same as yaw its in se3 space
  decomposed_tf.translation = se3Vec.head<3>();
  decomposed_tf.distance = decomposed_tf.translation.norm();
  // std::cout << "\n";
  // std::cout << "2,1_2,2:\t" << atan2(R(2,1),R(2,2))* (180/3.14159) << "\n";
  // std::cout << "1,0_0,0:\t" << atan2(R(1,0),R(0,0))* (180/3.14159) << "\n";
  // std::cout << "se3vec:\t" << se3Vec(5) * (180/3.14159) << "\n";
  // std::cout << "\n";
  return decomposed_tf;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluate the measurement error
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,1> TetherLengthErrorEval::evaluate() const {
  auto T_b_a = T_b_a_->evaluate();
  auto decomposed_tf = decomposeTF(T_b_a);
  double meas_distance_sq = pow(tether_meas_a_,2) + pow(tether_meas_b_,2);
  meas_distance_sq -= ( 2.0 * tether_meas_a_ * tether_meas_b_ * cos(decomposed_tf.yaw) );
  double meas_distance = sqrt(meas_distance_sq);

  Eigen::Matrix<double,1,1> result_error;
  result_error << (meas_distance - decomposed_tf.distance);

  return result_error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Interface for the general 'evaluation', with Jacobians
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,1> TetherLengthErrorEval::evaluate(const Eigen::Matrix<double,1,1>& lhs,
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

  auto decomposed_tf = decomposeTF(T_b_a);
  double meas_distance_sq = pow(tether_meas_a_,2) + pow(tether_meas_b_,2);
  meas_distance_sq -= ( 2.0 * tether_meas_a_ * tether_meas_b_ * cos(decomposed_tf.yaw) );
  double meas_distance = sqrt(meas_distance_sq);

  // Get Jacobians
  Eigen::Matrix<double,1,6> tether_err_jac = TetherModelJacobian(decomposed_tf);
  auto newLhs = lhs*tether_err_jac;
  T_b_a_->appendBlockAutomaticJacobians(newLhs, blkAutoEvalT_b_a.getRoot(), jacs);

  Eigen::Matrix<double,1,1> result_error;
  result_error << (meas_distance - decomposed_tf.distance);

  return result_error;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Error Jacobian (includes measured distance since it contains part of the state)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,6> TetherLengthErrorEval::TetherModelJacobian(DecomposedTF &decomposed_tf) const {
  // Precompute components
  auto &x = decomposed_tf.translation(0);
  auto &y = decomposed_tf.translation(1);
  auto &z = decomposed_tf.translation(2);
  auto &yaw = decomposed_tf.yaw;
  // auto &distance = decomposed_tf.distance;
  double meas_distance_sq = pow(tether_meas_a_,2) + pow(tether_meas_b_,2);
  meas_distance_sq -= ( 2.0 * tether_meas_a_ * tether_meas_b_ * cos(yaw) );
  double meas_distance = sqrt(meas_distance_sq);

  // Precompute Partial Derivatives (dx,dy,dz already are negative since it is a subtracted value)
  if(decomposed_tf.distance != 0)
  {
    dx = -x/decomposed_tf.distance;
    dy = -y/decomposed_tf.distance;
    dz = -z/decomposed_tf.distance;
    dr = (tether_meas_a_ * tether_meas_b_ * sin(yaw))/meas_distance;
  }

  // DEBUGGING: Print Statement
  // std::cout << "\n";
  // std::cout << "length_a:\t" << tether_meas_a_ << "\n";
  // std::cout << "length_b:\t" << tether_meas_b_ << "\n";
  // std::cout << "model_yaw:\t" <<  yaw*(180.0/3.1415) << "\n";
  // std::cout << "model_dist:\t" << decomposed_tf.distance << "\n";
  // std::cout << "meas_dist:\t" << meas_distance << "\n";
  // std::cout << "error_dist:\t" << (meas_distance - decomposed_tf.distance) << "\n";
  // std::cout << "dx/dy/dz/dr:\t" << dx << "/" << dy << "/" << dz << "/" << dr <<"\n";
  // std::cout << "\n";

  // Construct Jacobian with respect to x, y, z, and rotation (yaw)
  Eigen::Matrix<double,1,6> jac;
  jac << dx, dy, dz, 0.0, 0.0, dr;
  return jac;
}

} // steam
