//////////////////////////////////////////////////////////////////////////////////////////////
/// \file steam.hpp
/// \brief Convenience Header
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_ESTIMATION_LIBRARY_HPP
#define STEAM_ESTIMATION_LIBRARY_HPP

// base
#include <steam/OptimizationProblem.hpp>
#include <steam/CostTerm.hpp>
#include <steam/LossFunctions.hpp>
#include <steam/NoiseModel.hpp>

// blkmat
#include <steam/blockmat/BlockMatrix.hpp>
#include <steam/blockmat/BlockVector.hpp>
#include <steam/blockmat/BlockSparseMatrix.hpp>

// common
#include <steam/common/Time.hpp>

// evaluator
#include <steam/evaluator/ErrorEvaluator.hpp>
#include <steam/evaluator/TransformEvalOperations.hpp>
#include <steam/evaluator/TransformEvaluators.hpp>

// evaluator - common (sample functions)
#include <steam/evaluator/common/StereoCameraErrorEval.hpp>
#include <steam/evaluator/common/TransformErrorEval.hpp>
#include <steam/evaluator/common/VectorSpaceErrorEval.hpp>

// evaluator - jacobian
#include <steam/evaluator/jacobian/JacobianTreeBranchNode.hpp>
#include <steam/evaluator/jacobian/JacobianTreeLeafNode.hpp>

// solver
#include <steam/solver/VanillaGaussNewtonSolver.hpp>
#include <steam/solver/LineSearchGaussNewtonSolver.hpp>
#include <steam/solver/LevMarqGaussNewtonSolver.hpp>
#include <steam/solver/DoglegGaussNewtonSolver.hpp>

// state
#include <steam/state/StateVariable.hpp>
#include <steam/state/VectorSpaceStateVar.hpp>
#include <steam/state/LieGroupStateVar.hpp>
#include <steam/state/LandmarkStateVar.hpp>

// trajectory
#include <steam/trajectory/GpTrajectory.hpp>

#endif // STEAM_ESTIMATION_LIBRARY_HPP
