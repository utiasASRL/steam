#pragma once

#include <Eigen/Core>

#include "lgmath.hpp"

#include "steam/evaluable/evaluable.hpp"

namespace steam {
namespace stereo {

/** \brief Evaluator for the composition of a transformation and a landmark */
class ComposeLandmarkEvaluator : public Evaluable<Eigen::Vector4d> {
 public:
  using Ptr = std::shared_ptr<ComposeLandmarkEvaluator>;
  using ConstPtr = std::shared_ptr<const ComposeLandmarkEvaluator>;

  using PoseInType = lgmath::se3::Transformation;
  using LmInType = Eigen::Vector4d;
  using OutType = Eigen::Vector4d;

  static Ptr MakeShared(const Evaluable<PoseInType>::ConstPtr& transform,
                        const Evaluable<LmInType>::ConstPtr& landmark);
  ComposeLandmarkEvaluator(const Evaluable<PoseInType>::ConstPtr& transform,
                           const Evaluable<LmInType>::ConstPtr& landmark);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief Transform evaluable */
  const Evaluable<PoseInType>::ConstPtr transform_;
  /** \brief Landmark state variable */
  const Evaluable<LmInType>::ConstPtr landmark_;
};

ComposeLandmarkEvaluator::Ptr compose(
    const Evaluable<ComposeLandmarkEvaluator::PoseInType>::ConstPtr& transform,
    const Evaluable<ComposeLandmarkEvaluator::LmInType>::ConstPtr& landmark);

}  // namespace stereo
}  // namespace steam
