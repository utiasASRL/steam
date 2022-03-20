//////////////////////////////////////////////////////////////////////////////////////////////
/// \file steam.hpp
/// \brief Convenience Header
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

// blkmat
#include <steam/blockmat/BlockMatrix.hpp>
#include <steam/blockmat/BlockSparseMatrix.hpp>
#include <steam/blockmat/BlockVector.hpp>

// common
#include <steam/common/Time.hpp>
#include <steam/common/Timer.hpp>

// evaluable (including variable)
#include <steam/evaluable/evaluable.hpp>
#include <steam/evaluable/state_var.hpp>

// evaluable - se3
#include <steam/evaluable/se3/compose_evaluator.hpp>
#include <steam/evaluable/se3/inverse_evaluator.hpp>
#include <steam/evaluable/se3/log_map_evaluator.hpp>
#include <steam/evaluable/se3/se3_state_var.hpp>

// evaluable - vspace (vector space)
#include <steam/evaluable/vspace/vspace_state_var.hpp>

// problem
#include <steam/problem/LossFunctions.hpp>
#include <steam/problem/NoiseModel.hpp>
#include <steam/problem/OptimizationProblem.hpp>
#include <steam/problem/WeightedLeastSqCostTerm.hpp>

// solver
#include <steam/solver/DoglegGaussNewtonSolver.hpp>
#include <steam/solver/LevMarqGaussNewtonSolver.hpp>
#include <steam/solver/LineSearchGaussNewtonSolver.hpp>
#include <steam/solver/VanillaGaussNewtonSolver.hpp>

// trajectory
// #include <steam/trajectory/SteamTrajInterface.hpp>
// #include <steam/trajectory/SteamCATrajInterface.hpp>
