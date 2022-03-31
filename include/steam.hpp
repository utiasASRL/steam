//////////////////////////////////////////////////////////////////////////////////////////////
/// \file steam.hpp
/// \brief Convenience Header
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

// blkmat
#include "steam/blockmat/BlockMatrix.hpp"
#include "steam/blockmat/BlockSparseMatrix.hpp"
#include "steam/blockmat/BlockVector.hpp"

// common
#include "steam/common/Timer.hpp"

// evaluables (including variable)
#include "steam/evaluable/evaluable.hpp"
#include "steam/evaluable/state_var.hpp"

#include "steam/evaluable/p2p/p2p_error_evaluator.hpp"
#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/stereo/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"

// problem
#include "steam/problem/LossFunctions.hpp"
#include "steam/problem/NoiseModel.hpp"
#include "steam/problem/OptimizationProblem.hpp"
#include "steam/problem/WeightedLeastSqCostTerm.hpp"

// solver
#include "steam/solver/DoglegGaussNewtonSolver.hpp"
#include "steam/solver/LevMarqGaussNewtonSolver.hpp"
#include "steam/solver/LineSearchGaussNewtonSolver.hpp"
#include "steam/solver/VanillaGaussNewtonSolver.hpp"

// trajectory
#include "steam/trajectory/const_vel/interface.hpp"
