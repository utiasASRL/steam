#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class P2PErrorDopplerEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<P2PErrorDopplerEvaluator>;
  using ConstPtr = std::shared_ptr<const P2PErrorDopplerEvaluator>;

  using PoseInType = lgmath::se3::Transformation;
  using VelInType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &T_rq,
                        const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                        const Eigen::Vector3d &reference,
                        const Eigen::Vector3d &query, const float beta) {
      return MakeShared(T_rq, w_r_q_in_q, reference, query, beta, false);
    }
  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr &T_rq,
                        const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                        const Eigen::Vector3d &reference,
                        const Eigen::Vector3d &query, const float beta,
                        const bool rm_ori);

  P2PErrorDopplerEvaluator(const Evaluable<PoseInType>::ConstPtr &T_rq,
                          const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                          const Eigen::Vector3d &reference,
                          const Eigen::Vector3d &query, const float beta)
      : P2PErrorDopplerEvaluator(T_rq, w_r_q_in_q, reference, query, beta, false) {}
  P2PErrorDopplerEvaluator(const Evaluable<PoseInType>::ConstPtr &T_rq,
                           const Evaluable<VelInType>::ConstPtr &w_r_q_in_q,
                           const Eigen::Vector3d &reference,
                           const Eigen::Vector3d &query, const float beta,
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
  Eigen::Matrix<double, 3, 4> D_ = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Vector4d reference_ = Eigen::Vector4d::Constant(1);
  Eigen::Vector4d query_ = Eigen::Vector4d::Constant(1);
  const float beta_;
  const bool rm_ori_;
};

P2PErrorDopplerEvaluator::Ptr p2pErrorDoppler(
    const Evaluable<P2PErrorDopplerEvaluator::PoseInType>::ConstPtr &T_rq,
    const Evaluable<P2PErrorDopplerEvaluator::VelInType>::ConstPtr &w_r_q_in_q,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query,
    const float beta, const bool rm_ori);

inline P2PErrorDopplerEvaluator::Ptr p2pErrorDoppler(
  const Evaluable<P2PErrorDopplerEvaluator::PoseInType>::ConstPtr &T_rq,
  const Evaluable<P2PErrorDopplerEvaluator::VelInType>::ConstPtr &w_r_q_in_q,
  const Eigen::Vector3d &reference, const Eigen::Vector3d &query,
  const float beta) {
    return p2pErrorDoppler(T_rq, w_r_q_in_q, reference, query, beta, false);
  }

}  // namespace p2p
}  // namespace steam