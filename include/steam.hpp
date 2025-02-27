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

#include "steam/evaluable/imu/evaluables.hpp"
#include "steam/evaluable/p2p/evaluables.hpp"
#include "steam/evaluable/se3/evaluables.hpp"
#include "steam/evaluable/stereo/evaluables.hpp"
#include "steam/evaluable/vspace/evaluables.hpp"

// problem
#include "steam/problem/cost_term/weighted_least_sq_cost_term.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/noise_model/static_noise_model.hpp"
#include "steam/problem/noise_model/dynamic_noise_model.hpp"
#include "steam/problem/optimization_problem.hpp"
#include "steam/problem/sliding_window_filter.hpp"

// solver
#include "steam/solver/covariance.hpp"
#include "steam/solver/dogleg_gauss_newton_solver.hpp"
#include "steam/solver/gauss_newton_solver.hpp"
#include "steam/solver/gauss_newton_solver_nva.hpp"
#include "steam/solver/lev_marq_gauss_newton_solver.hpp"
#include "steam/solver/line_search_gauss_newton_solver.hpp"

// trajectory
#include "steam/trajectory/bspline/interface.hpp"
#include "steam/trajectory/const_acc/interface.hpp"
#include "steam/trajectory/const_vel/interface.hpp"
#include "steam/trajectory/singer/interface.hpp"
