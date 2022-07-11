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

#include "steam/evaluable/p2p/evaluables.hpp"
#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/stereo/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"

// problem
#include "steam/problem/OptimizationProblem.hpp"
#include "steam/problem/cost_term/weighted_least_sq_cost_term.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/noise_model/static_noise_model.hpp"

// solver
#include "steam/solver/DoglegGaussNewtonSolver.hpp"
#include "steam/solver/LevMarqGaussNewtonSolver.hpp"
#include "steam/solver/LineSearchGaussNewtonSolver.hpp"
#include "steam/solver/VanillaGaussNewtonSolver.hpp"

// trajectory
#include "steam/trajectory/const_vel/interface.hpp"
#include "steam/trajectory/bspline/interface.hpp"
