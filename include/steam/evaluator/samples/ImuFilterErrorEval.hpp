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

  ImuFilterErrorEval(const std::vector<double> & imu_meas_rpy_a,
                     const std::vector<double> & imu_meas_rpy_b,
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

private:
  ComponentsTF componentTF(lgmath::se3::Transformation &tf) const;
  std::vector<double> imu_meas_rpy_a_;
  std::vector<double> imu_meas_rpy_b_;
  se3::TransformEvaluator::ConstPtr T_b_a_;

};

} //steam

#endif //
