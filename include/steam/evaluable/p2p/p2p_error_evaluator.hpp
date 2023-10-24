#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace p2p {

class P2PErrorEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<P2PErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const P2PErrorEvaluator>;

  using InType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 3, 1>;
  using Time = steam::traj::Time;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                        const Eigen::Vector3d &reference,
                        const Eigen::Vector3d &query);
  P2PErrorEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                    const Eigen::Vector3d &reference,
                    const Eigen::Vector3d &query);

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

  Eigen::Matrix<double, 3, 6> getJacobianPose() const;

 private:
  // evaluable
  const Evaluable<InType>::ConstPtr T_rq_;
  // constants
  Eigen::Matrix<double, 3, 4> D_ = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Vector4d reference_ = Eigen::Vector4d::Constant(1);
  Eigen::Vector4d query_ = Eigen::Vector4d::Constant(1);
  bool time_init_ = false;
  Time time_;
};

P2PErrorEvaluator::Ptr p2pError(
    const Evaluable<P2PErrorEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query);

}  // namespace p2p
}  // namespace steam