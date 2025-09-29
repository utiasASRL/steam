#pragma once

#include <Eigen/Core>
#include <cmath>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace p2p {

/** \brief Evaluator for 2D bearing-range measurements. Assumes that the vehicle
 * and the landmark are close to the z=0 plane */
class BRError2DEvaluator : public Evaluable<Eigen::Matrix<double, 2, 1>> {
 public:
  using Ptr = std::shared_ptr<BRError2DEvaluator>;
  using ConstPtr = std::shared_ptr<const BRError2DEvaluator>;

  using InType = Eigen::Vector4d;
  using OutType = Eigen::Matrix<double, 2, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& point,
                        const Eigen::Vector2d& br_meas);

  BRError2DEvaluator(const Evaluable<InType>::ConstPtr& point,
                     const Eigen::Vector2d& br_meas);

  bool active() const override;

  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;

  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief point evaluable */
  const Evaluable<InType>::ConstPtr point_;
  const Eigen::Vector2d br_meas_;
  OutType br_val_ = Eigen::Vector2d(0.0, 0.0);
};

BRError2DEvaluator::Ptr br2dError(
    const Evaluable<BRError2DEvaluator::InType>::ConstPtr& point,
    const Eigen::Vector2d& br_meas);

}  // namespace p2p
}  // namespace steam