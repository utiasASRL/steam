//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvalOperations.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_TRANSFORM_EVALUATOR_OPERATIONS_HPP
#define STEAM_TRANSFORM_EVALUATOR_OPERATIONS_HPP

#include <steam/evaluator/blockauto/transform/ComposeTransformEvaluator.hpp>
#include <steam/evaluator/blockauto/transform/InverseTransformEvaluator.hpp>
#include <steam/evaluator/blockauto/transform/ComposeLandmarkEvaluator.hpp>
#include <steam/evaluator/blockauto/transform/LogMapEvaluator.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compose two transform evaluators
//////////////////////////////////////////////////////////////////////////////////////////////
static TransformEvaluator::Ptr compose(const TransformEvaluator::ConstPtr& transform1,
                                       const TransformEvaluator::ConstPtr& transform2) {
  return ComposeTransformEvaluator::MakeShared(transform1, transform2);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compose a transform evaluator and landmark state variable
//////////////////////////////////////////////////////////////////////////////////////////////
static ComposeLandmarkEvaluator::Ptr compose(const TransformEvaluator::ConstPtr& transform,
                                             const se3::LandmarkStateVar::Ptr& landmark) {
  return ComposeLandmarkEvaluator::MakeShared(transform, landmark);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Invert a transform evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
static TransformEvaluator::Ptr inverse(const TransformEvaluator::ConstPtr& transform) {
  return InverseTransformEvaluator::MakeShared(transform);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Take the 'logarithmic map' of a transformation evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
static LogMapEvaluator::Ptr tran2vec(const TransformEvaluator::ConstPtr& transform) {
  return LogMapEvaluator::MakeShared(transform);
}

} // se3
} // steam

#endif // STEAM_TRANSFORM_EVALUATOR_OPERATIONS_HPP
