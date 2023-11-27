#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {
namespace imu {

template <int DIM = Eigen::Dynamic>
class VSpaceInterpolator : public Evaluable<Eigen::Matrix<double, DIM, 1>> {
 public:
  using Ptr = std::shared_ptr<VSpaceInterpolator>;
  using ConstPtr = std::shared_ptr<const VSpaceInterpolator>;

  using InType = Eigen::Matrix<double, DIM, 1>;
  using OutType = Eigen::Matrix<double, DIM, 1>;
  using Time = steam::traj::Time;

  static Ptr MakeShared(const Time& time,
                        const typename Evaluable<InType>::ConstPtr& bias1,
                        const Time& time1,
                        const typename Evaluable<InType>::ConstPtr& bias2,
                        const Time& time2);
  VSpaceInterpolator(const Time& time,
                     const typename Evaluable<InType>::ConstPtr& bias1,
                     const Time& time1,
                     const typename Evaluable<InType>::ConstPtr& bias2,
                     const Time& time2);
  bool active() const override;
  using KeySet = typename Evaluable<OutType>::KeySet;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  typename Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs,
                const typename Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const typename Evaluable<InType>::ConstPtr bias1_;
  const typename Evaluable<InType>::ConstPtr bias2_;
  double psi_, lambda_;
};

template <int DIM>
auto VSpaceInterpolator<DIM>::MakeShared(
    const Time& time, const typename Evaluable<InType>::ConstPtr& bias1,
    const Time& time1, const typename Evaluable<InType>::ConstPtr& bias2,
    const Time& time2) -> Ptr {
  return std::make_shared<VSpaceInterpolator>(time, bias1, time1, bias2, time2);
}

template <int DIM>
VSpaceInterpolator<DIM>::VSpaceInterpolator(
    const Time& time, const typename Evaluable<InType>::ConstPtr& bias1,
    const Time& time1, const typename Evaluable<InType>::ConstPtr& bias2,
    const Time& time2)
    : bias1_(bias1), bias2_(bias2) {
  if (time < time1 || time > time2)
    throw std::runtime_error("time < time1 || time > time2");
  const double tau = (time - time1).seconds();
  const double T = (time2 - time1).seconds();
  const double ratio = tau / T;
  psi_ = ratio;
  lambda_ = 1 - ratio;
}

template <int DIM>
bool VSpaceInterpolator<DIM>::active() const {
  return bias1_->active() || bias2_->active();
}

template <int DIM>
void VSpaceInterpolator<DIM>::getRelatedVarKeys(KeySet& keys) const {
  bias1_->getRelatedVarKeys(keys);
  bias2_->getRelatedVarKeys(keys);
}

template <int DIM>
auto VSpaceInterpolator<DIM>::value() const -> OutType {
  return lambda_ * bias1_->value() + psi_ * bias2_->value();
}

template <int DIM>
auto VSpaceInterpolator<DIM>::forward() const -> typename Node<OutType>::Ptr {
  const auto b1 = bias1_->forward();
  const auto b2 = bias2_->forward();
  OutType b = lambda_ * b1->value() + psi_ * b2->value();
  const auto node = Node<OutType>::MakeShared(b);
  node->addChild(b1);
  node->addChild(b2);
  return node;
}

template <int DIM>
void VSpaceInterpolator<DIM>::backward(const Eigen::MatrixXd& lhs,
                                       const typename Node<OutType>::Ptr& node,
                                       Jacobians& jacs) const {
  if (!active()) return;
  if (bias1_->active()) {
    const auto b1_ = std::static_pointer_cast<Node<InType>>(node->at(0));
    bias1_->backward(lambda_ * lhs, b1_, jacs);
  }
  if (bias2_->active()) {
    const auto b2_ = std::static_pointer_cast<Node<InType>>(node->at(1));
    bias2_->backward(psi_ * lhs, b2_, jacs);
  }
}

}  // namespace imu
}  // namespace steam