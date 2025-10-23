#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class P2PSE2ErrorDopplerEvaluator : public Evaluable<Eigen::Matrix<double, 2, 1>> {
 public:
  using Ptr = std::shared_ptr<P2PSE2ErrorDopplerEvaluator>;
  using ConstPtr = std::shared_ptr<const P2PSE2ErrorDopplerEvaluator>;

  using PoseInType = lgmath::se2::Transformation;
  using VelInType = Eigen::Matrix<double, 3, 1>;
  using OutType = Eigen::Matrix<double, 2, 1>;

  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &T_rq,
                        const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                        const Eigen::Vector2d &reference,
                        const Eigen::Vector2d &query, const float beta) {
      return MakeShared(T_rq, w_r_q_in_q, reference, query, beta, false);
    }
  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &T_rq,
                        const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                        const Eigen::Vector2d &reference,
                        const Eigen::Vector2d &query, const float beta,
                        const bool rm_ori);

  P2PSE2ErrorDopplerEvaluator(const Evaluable<PoseInType>::ConstPtr &T_rq,
                          const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                          const Eigen::Vector2d &reference,
                          const Eigen::Vector2d &query, const float beta)
      : P2PSE2ErrorDopplerEvaluator(T_rq, w_r_q_in_q, reference, query, beta, false) {}
  P2PSE2ErrorDopplerEvaluator(const Evaluable<PoseInType>::ConstPtr &T_rq,
                           const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                           const Eigen::Vector2d &reference,
                           const Eigen::Vector2d &query, const float beta,
                           const bool rm_ori);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

 private:
  // evaluable
  const Evaluable<PoseInType>::ConstPtr T_rq_;
  const Evaluable<VelInType>::ConstPtr w_r_q_in_q_;
  // constants
  Eigen::Matrix<double, 2, 3> D_ = Eigen::Matrix<double, 2, 3>::Zero();
  Eigen::Vector3d reference_ = Eigen::Vector3d::Constant(1);
  Eigen::Vector3d query_ = Eigen::Vector3d::Constant(1);
  const float beta_;
  const bool rm_ori_;
};

P2PSE2ErrorDopplerEvaluator::Ptr p2pSE2ErrorDoppler(
    const Evaluable<P2PSE2ErrorDopplerEvaluator::PoseInType>::ConstPtr &T_rq,
    const Evaluable<P2PSE2ErrorDopplerEvaluator::VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector2d &reference, const Eigen::Vector2d &query,
    const float beta, const bool rm_ori);

inline P2PSE2ErrorDopplerEvaluator::Ptr p2pSE2ErrorDoppler(
  const Evaluable<P2PSE2ErrorDopplerEvaluator::PoseInType>::ConstPtr &T_rq,
  const Evaluable<P2PSE2ErrorDopplerEvaluator::VelInType>::ConstPtr &w_r_q_in_q,
  const Eigen::Vector2d &reference, const Eigen::Vector2d &query,
  const float beta) {
    return p2pSE2ErrorDoppler(T_rq, w_r_q_in_q, reference, query, beta, false);
  }

}  // namespace p2p
}  // namespace steam