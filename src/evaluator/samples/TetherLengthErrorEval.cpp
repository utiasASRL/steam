//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TetherLengthErrorEval.cpp
///
/// \author Patrick McGarey, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/evaluator/samples/TetherLengthErrorEval.hpp>

namespace steam {

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
/// \brief Returns whether or not an evaluator contains unlocked state variables
//////////////////////////////////////////////////////////////////////////////////////////////
bool TetherLengthErrorEval::isActive() const {
  return T_b_a_->isActive();
}

DecomposedTF TetherLengthErrorEval::decomposeTF(lgmath::se3::Transformation &tf) const {
  DecomposedTF decomposed_tf;
  auto se3Vec = tf.vec();
  decomposed_tf.yaw =  se3Vec(5);
  decomposed_tf.translation = se3Vec.head<3>();
  decomposed_tf.distance = decomposed_tf.translation.norm();
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

  return Eigen::Matrix<double,1,1>(meas_distance - decomposed_tf.distance);
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
  auto newLhs = -1*lhs*tether_err_jac;
  T_b_a_->appendBlockAutomaticJacobians(newLhs, blkAutoEvalT_b_a.getRoot(), jacs);
  
  // Return Evaluation
  return Eigen::Matrix<double,1,1>(meas_distance - decomposed_tf.distance);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Error Model Jacobian
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<double,1,6> TetherLengthErrorEval::TetherModelJacobian(DecomposedTF &decomposed_tf) const {
  // Precompute components
  auto &x = decomposed_tf.translation(0);
  auto &y = decomposed_tf.translation(1);
  auto &z = decomposed_tf.translation(2);
  // TODO this is a repreat of above do we need to call this again
  auto &yaw = decomposed_tf.yaw;
  auto &distance = decomposed_tf.distance;
  double meas_distance_sq = pow(tether_meas_a_,2) + pow(tether_meas_b_,2);
  meas_distance_sq -= ( 2.0 * tether_meas_a_ * tether_meas_b_ * cos(yaw) );
  double meas_distance = sqrt(meas_distance_sq);
  
  // Precompute Partial Derivatives
  double dx = -x/distance;
  double dy = -y/distance;
  double dz = -z/distance;
  double dr = (x * y * sin(yaw))/meas_distance;

  // TODO Construct Jacobian with respect to x, y, z, and rotation (yaw)
  Eigen::Matrix<double,1,6> jac;
  jac << dx, dy, dz, 0.0, 0.0, dr;
  return jac;
}

} // steam
