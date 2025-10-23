#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace p2p {

class P2PSE2ErrorEvaluator : public Evaluable<Eigen::Matrix<double, 2, 1>> {
 public:
  using Ptr = std::shared_ptr<P2PSE2ErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const P2PSE2ErrorEvaluator>;

  using InType = lgmath::se2::Transformation;
  using OutType = Eigen::Matrix<double, 2, 1>;
  using Time = steam::traj::Time;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                        const Eigen::Vector2d &reference,
                        const Eigen::Vector2d &query) {
    return MakeShared(T_rq, reference, query, false);
  }
  static Ptr MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                        const Eigen::Vector2d &reference,
                        const Eigen::Vector2d &query,
                        const bool rm_ori);

  P2PSE2ErrorEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                    const Eigen::Vector2d &reference,
                    const Eigen::Vector2d &query)
      : P2PSE2ErrorEvaluator(T_rq, reference, query, false) {}
  P2PSE2ErrorEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                    const Eigen::Vector2d &reference,
                    const Eigen::Vector2d &query,
                    const bool rm_ori);

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

  Eigen::Matrix<double, 2, 3> getJacobianPose() const;

 private:
  // evaluable
  const Evaluable<InType>::ConstPtr T_rq_;
  // constants
  Eigen::Matrix<double, 2, 3> D_ = Eigen::Matrix<double, 2, 3>::Zero();
  Eigen::Vector3d reference_ = Eigen::Vector3d::Constant(1);
  Eigen::Vector3d query_ = Eigen::Vector3d::Constant(1);
  bool time_init_ = false;
  bool rm_ori_;
  Time time_;
};

P2PSE2ErrorEvaluator::Ptr p2pSE2Error(
    const Evaluable<P2PSE2ErrorEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector2d &reference, const Eigen::Vector2d &query,
    const bool rm_ori);

inline P2PSE2ErrorEvaluator::Ptr p2pSE2Error(
    const Evaluable<P2PSE2ErrorEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector2d &reference,
    const Eigen::Vector2d &query) {
      return p2pSE2Error(T_rq, reference, query, false);
    }

}  // namespace p2p
}  // namespace steam