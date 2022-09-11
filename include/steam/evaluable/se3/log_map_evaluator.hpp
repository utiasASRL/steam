#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace se3 {

class LogMapEvaluator : public Evaluable<Eigen::Matrix<double, 6, 1>> {
 public:
  using Ptr = std::shared_ptr<LogMapEvaluator>;
  using ConstPtr = std::shared_ptr<const LogMapEvaluator>;

  using InType = lgmath::se3::Transformation;
  using OutType = Eigen::Matrix<double, 6, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& transform);
  LogMapEvaluator(const Evaluable<InType>::ConstPtr& transform);

  bool active() const override;
  void getRelatedVarKeys(KeySet &keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  const Evaluable<InType>::ConstPtr transform_;
};

LogMapEvaluator::Ptr tran2vec(
    const Evaluable<LogMapEvaluator::InType>::ConstPtr& transform);

}  // namespace se3
}  // namespace steam