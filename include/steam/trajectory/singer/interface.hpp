#pragma once

#include <Eigen/Core>

#include "steam/trajectory/const_acc/interface.hpp"
#include "steam/trajectory/time.hpp"

#include "steam/trajectory/singer/acceleration_extrapolator.hpp"
#include "steam/trajectory/singer/acceleration_interpolator.hpp"
#include "steam/trajectory/singer/helper.hpp"
#include "steam/trajectory/singer/pose_extrapolator.hpp"
#include "steam/trajectory/singer/pose_interpolator.hpp"
#include "steam/trajectory/singer/prior_factor.hpp"
#include "steam/trajectory/singer/velocity_extrapolator.hpp"
#include "steam/trajectory/singer/velocity_interpolator.hpp"

namespace steam {
namespace traj {
namespace singer {

class Interface : public steam::traj::const_acc::Interface {
 public:
  using Ptr = std::shared_ptr<Interface>;
  using ConstPtr = std::shared_ptr<const Interface>;
  using Variable = steam::traj::const_acc::Variable;

  using PoseType = lgmath::se3::Transformation;
  using VelocityType = Eigen::Matrix<double, 6, 1>;
  using AccelerationType = Eigen::Matrix<double, 6, 1>;
  // using CovType = Eigen::Matrix<double, 18, 18>;

  static Ptr MakeShared(const Eigen::Matrix<double, 6, 1>& alpha_diag =
                            Eigen::Matrix<double, 6, 1>::Ones(),
                        const Eigen::Matrix<double, 6, 1>& Qc_diag =
                            Eigen::Matrix<double, 6, 1>::Ones()) {
    return std::make_shared<Interface>(alpha_diag, Qc_diag);
  }

  Interface(const Eigen::Matrix<double, 6, 1>& alpha_diag =
                Eigen::Matrix<double, 6, 1>::Ones(),
            const Eigen::Matrix<double, 6, 1>& Qc_diag =
                Eigen::Matrix<double, 6, 1>::Ones())
      : steam::traj::const_acc::Interface(Qc_diag), alpha_diag_(alpha_diag) {}

 protected:
  Eigen::Matrix<double, 6, 1> alpha_diag_;
  Eigen::Matrix<double, 18, 18> getJacKnot1_(
      const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
    return getJacKnot1(knot1, knot2, alpha_diag_);
  }
  Eigen::Matrix<double, 18, 18> getQ_(
      const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
    return getQ(dt, alpha_diag_, Qc_diag);
  }
  Eigen::Matrix<double, 18, 18> getQinv_(
      const double& dt, const Eigen::Matrix<double, 6, 1>& Qc_diag) const {
    return getQ(dt, alpha_diag_, Qc_diag).inverse();
  }
  Evaluable<PoseType>::Ptr getPoseInterpolator_(
      const Time& time, const Variable::ConstPtr& knot1,
      const Variable::ConstPtr& knot2) const {
    return PoseInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
  }
  Evaluable<VelocityType>::Ptr getVelocityInterpolator_(
      const Time& time, const Variable::ConstPtr& knot1,
      const Variable::ConstPtr& knot2) const {
    return VelocityInterpolator::MakeShared(time, knot1, knot2, alpha_diag_);
  }
  Evaluable<AccelerationType>::Ptr getAccelerationInterpolator_(
      const Time& time, const Variable::ConstPtr& knot1,
      const Variable::ConstPtr& knot2) const {
    return AccelerationInterpolator::MakeShared(time, knot1, knot2,
                                                alpha_diag_);
  }
  Evaluable<PoseType>::Ptr getPoseExtrapolator_(
      const Time& time, const Variable::ConstPtr& knot) const {
    return PoseExtrapolator::MakeShared(time, knot, alpha_diag_);
  }
  Evaluable<VelocityType>::Ptr getVelocityExtrapolator_(
      const Time& time, const Variable::ConstPtr& knot) const {
    return VelocityExtrapolator::MakeShared(time, knot, alpha_diag_);
  }
  Evaluable<AccelerationType>::Ptr getAccelerationExtrapolator_(
      const Time& time, const Variable::ConstPtr& knot) const {
    return AccelerationExtrapolator::MakeShared(time, knot, alpha_diag_);
  }
  Evaluable<Eigen::Matrix<double, 18, 1>>::Ptr getPriorFactor_(
      const Variable::ConstPtr& knot1, const Variable::ConstPtr& knot2) const {
    return PriorFactor::MakeShared(knot1, knot2, alpha_diag_);
  }
};

}  // namespace singer
}  // namespace traj
}  // namespace steam