#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace p2p {

class P2PlaneErrorEvaluator : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<P2PlaneErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const P2PlaneErrorEvaluator>;

  using InType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 1, 1>;
  using Time = steam::traj::Time;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                        const Eigen::Vector3d &reference,
                        const Eigen::Vector3d &query,
                        const Eigen::Vector3d &normal);
  P2PlaneErrorEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                    const Eigen::Vector3d &reference,
                    const Eigen::Vector3d &query,
                    const Eigen::Vector3d &normal);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

  void setTime(Time time) {
    time_ = time;
    time_init_ = true;
  };
  Time getTime() const {
    if (time_init_)
      return time_;
    else
      throw std::runtime_error("P2P measurement time was not initialized");
  }

  Eigen::Matrix<double, 1, 6> getJacobianPose() const;

 private:
  // evaluable
  const Evaluable<InType>::ConstPtr T_rq_;
  // constants
  const Eigen::Vector3d reference_;
  const Eigen::Vector3d query_;
  const Eigen::Vector3d normal_;
  
  bool time_init_ = false;
  Time time_;
};

P2PlaneErrorEvaluator::Ptr p2planeError(
    const Evaluable<P2PlaneErrorEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &, const Eigen::Vector3d &normal);

}  // namespace p2p
}  // namespace steam