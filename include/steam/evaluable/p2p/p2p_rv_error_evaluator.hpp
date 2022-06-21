#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

class P2PRVErrorEvaluator : public Evaluable<Eigen::Matrix<double, 4, 1>> {
 public:
  using Ptr = std::shared_ptr<P2PRVErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const P2PRVErrorEvaluator>;

  using InP2PType = Eigen::Matrix<double, 3, 1>;
  using InRVType = Eigen::Matrix<double, 1, 1>;
  using OutType = Eigen::Matrix<double, 4, 1>;

  static Ptr MakeShared(const Evaluable<InP2PType>::ConstPtr &p2p,
                        const Evaluable<InRVType>::ConstPtr &rv);
  P2PRVErrorEvaluator(const Evaluable<InP2PType>::ConstPtr &p2p,
                      const Evaluable<InRVType>::ConstPtr &rv);

  bool active() const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd &lhs, const Node<OutType>::Ptr &node,
                Jacobians &jacs) const override;

 private:
  // evaluable
  const Evaluable<InP2PType>::ConstPtr p2p_;
  const Evaluable<InRVType>::ConstPtr rv_;
};

P2PRVErrorEvaluator::Ptr p2prvError(
    const Evaluable<P2PRVErrorEvaluator::InP2PType>::ConstPtr &p2p,
    const Evaluable<P2PRVErrorEvaluator::InRVType>::ConstPtr &rv);

}  // namespace p2p
}  // namespace steam