#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class P2PErrorWithDopplerCompensationEvaluator : public Evaluable<Eigen::Matrix<double, 3, 1>> {
 public:
  using Ptr = std::shared_ptr<P2PErrorWithDopplerCompensationEvaluator>;
  using ConstPtr = std::shared_ptr<const P2PErrorWithDopplerCompensationEvaluator>;

  using InType = lgmath::se3::Transformation;
  using VType = Eigen::Matrix<double, 6, 1>;
  using OutType = Eigen::Matrix<double, 3, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr &T_rq,
                        const Eigen::Vector3d &reference,
                        const Eigen::Vector3d &query,
                        const float beta,
                        const Evaluable<VType>::ConstPtr &w_r_q_in_q);
  P2PErrorWithDopplerCompensationEvaluator(const Evaluable<InType>::ConstPtr &T_rq,
                    const Eigen::Vector3d &reference,
                    const Eigen::Vector3d &query,
                    const float beta,
                    const Evaluable<VType>::ConstPtr &w_r_q_in_q);

  bool active() const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

 private:
  // evaluable
  const Evaluable<InType>::ConstPtr T_rq_;
  // constants
  Eigen::Matrix<double, 3, 4> D_ = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Vector4d reference_ = Eigen::Vector4d::Constant(1);
  Eigen::Vector4d query_ = Eigen::Vector4d::Constant(1);
  float beta_ = 0.049;
  // evaluable
  const Evaluable<VType>::ConstPtr w_r_q_in_q_;
};

P2PErrorWithDopplerCompensationEvaluator::Ptr p2pError(
    const Evaluable<P2PErrorWithDopplerCompensationEvaluator::InType>::ConstPtr &T_rq,
    const Eigen::Vector3d &reference, const Eigen::Vector3d &query, const float beta,
    const Evaluable<P2PErrorWithDopplerCompensationEvaluator::VType>::ConstPtr &w_r_q_in_q);

}  // namespace p2p
}  // namespace steam