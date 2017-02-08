//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TetherLengthErrorEval.cpp
///
/// \author Patrick McGarey, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_TETHER_LENGTH_ERROR_EVALUATOR_HPP
#define STEAM_TETHER_LENGTH_ERROR_EVALUATOR_HPP

// #pragma once

#include <steam.hpp>

namespace steam {

struct DecomposedTF {
  double yaw;
  double distance;
  Eigen::Vector3d translation;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief tether length error function evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
class TetherLengthErrorEval : public ErrorEvaluator<1,6>::type
{
public:
  /// Convenience typedefs
  typedef boost::shared_ptr<TetherLengthErrorEval> Ptr;
  typedef boost::shared_ptr<const TetherLengthErrorEval> ConstPtr;

  TetherLengthErrorEval(const double & tether_length_a,
                        const double & tether_length_b,
                        const se3::TransformEvaluator::ConstPtr& T_b_a);

  virtual ~TetherLengthErrorEval();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Interface for the general 'evaluation'
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Matrix<double,1,1> evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Interface for the general 'evaluation', with Jacobians
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Matrix<double,1,1> evaluate(const Eigen::Matrix<double,1,1>& lhs,
                        std::vector<Jacobian<1,6> >* jacs) const;

  Eigen::Matrix<double,1,6> TetherModelJacobian(DecomposedTF &decomposed_tf) const;

private:
  DecomposedTF decomposeTF(lgmath::se3::Transformation &tf) const;
  double tether_meas_a_;
  double tether_meas_b_;
  se3::TransformEvaluator::ConstPtr T_b_a_;

};

} //steam

#endif //
