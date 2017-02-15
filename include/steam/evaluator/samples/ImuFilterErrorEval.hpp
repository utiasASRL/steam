//////////////////////////////////////////////////////////////////////////////////////////////
/// \file ImuFilterErrorEval.cpp
///
/// \author Patrick McGarey, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

//TODO: get correct size for Eigen matrices below

#ifndef IMU_FILTER_LENGTH_ERROR_EVALUATOR_HPP
#define IMU_FILTER_LENGTH_ERROR_EVALUATOR_HPP

// #pragma once

#include <steam.hpp>

namespace steam {

struct ComponentsTF {
  double roll;
  double pitch;
  double yaw;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief tether length error function evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
class ImuFilterErrorEval : public ErrorEvaluator<1,6>::type
{
public:
  /// Convenience typedefs
  typedef boost::shared_ptr<ImuFilterErrorEval> Ptr;
  typedef boost::shared_ptr<const ImuFilterErrorEval> ConstPtr;

  ImuFilterErrorEval(const double & imu_meas_roll_,
                        const double & imu_meas_pitch_,
                        const double & imu_meas_yaw_,
                        const se3::TransformEvaluator::ConstPtr& T_b_a);

  virtual ~ImuFilterErrorEval();

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

  Eigen::Matrix<double,1,6> TetherModelJacobian(ComponentsTF &decomposed_tf) const;

private:
  ComponentsTF componentTF(lgmath::se3::Transformation &tf) const;
  double imu_meas_roll_;
  double imu_meas_pitch_;
  double imu_meas_yaw_;
  se3::TransformEvaluator::ConstPtr T_b_a_;

};

} //steam

#endif //
