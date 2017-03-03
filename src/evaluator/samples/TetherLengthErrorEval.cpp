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

  // first get rotation, then convert from lg to eigen
  Eigen::Matrix3d R = tf.C_ba().matrix();

  double yaw;

  // check if singular
  double sy = sqrt(R(0,0) * R(0,0) +  R(1,0) * R(1,0));

  if (sy < 1e-6) {
    yaw = 0;
    std::cout << "WARN: singularity, setting yaw = 0" << "\n";
  }
  else {
    yaw = atan2(R(1,0),R(0,0));
  }

  decomposed_tf.yaw =  yaw;
  decomposed_tf.translation = tf.r_ab_inb(); // this is similar to se3
  decomposed_tf.distance = decomposed_tf.translation.norm();

  // Uses se3vec function which causes inaccuracies
  // auto se3Vec = tf.vec();
  // auto se3_yaw =  se3Vec(5);
  // auto se3_translation = se3Vec.head<3>();
  // auto se3_distance = se3_translation.norm();
  // decomposed_tf.yaw =  se3_yaw;
  // decomposed_tf.translation = se3_translation;
  // decomposed_tf.distance = se3_distance;

  //print statements (debugging)
  // std::cout << "\n";
  // std::cout << "yaw_ba:\t" << decomposed_tf.yaw * (180/3.14159) << "\n";
  // std::cout << "se3yaw:\t" << se3_yaw * (180/3.14159) << "\n";
  // std::cout << "\n";
  // std::cout << "d:\t" << decomposed_tf.distance << "\n";
  // std::cout << "d_se3:\t" << se3_distance << "\n";
  // std::cout << "\n";
  // std::cout << "x_ab:\t" << decomposed_tf.translation(0) << "\n";
  // std::cout << "x_se3:\t" << se3_translation(0) << "\n";
  // std::cout << "\n";
  // std::cout << "y_ab:\t" << decomposed_tf.translation(1) << "\n";
  // std::cout << "y_se3:\t" << se3_translation(1) << "\n";
  // std::cout << "\n";
  // std::cout << "z_ab:\t" << decomposed_tf.translation(2) << "\n";
  // std::cout << "z_se3:\t" << se3_translation(2) << "\n";
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
