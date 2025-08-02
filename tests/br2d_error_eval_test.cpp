#include <gtest/gtest.h>

#include <iostream>
#include <random>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace std;
using namespace steam;
using namespace steam::traj;
using namespace steam::p2p;

#define EXPECT_MATRIX_NEAR(A, B, tol)                         \
  EXPECT_TRUE((A).isApprox((B), tol)) << "Expected:\n"        \
                                      << (A) << "\nActual:\n" \
                                      << (B)

// Test constructor
TEST(BearingRange2DEval, Constructor) {
  const Eigen::Vector4d lm_vec(1.0, 0.0, 0.0, 1.0);
  const auto landmark = vspace::VSpaceStateVar<4>::MakeShared(lm_vec);
  BRError2DEvaluator(landmark, Eigen::Vector2d(0.0, 0.0));
  BRError2DEvaluator::MakeShared(landmark, Eigen::Vector2d(0.0, 0.0));
}

// test error evaluation
TEST(BearingRange2DEval, ErrEval) {
  double tol = 1e-6;
  // landmark at x=1 y=1, NOTE: Z component is ignored
  auto landmark = vspace::VSpaceStateVar<4>::MakeShared(
      Eigen::Vector4d(1.0, 1.0, 1.0, 1.0));
  auto meas = Eigen::Vector2d(M_PI / 4, sqrt(2));
  auto err_func = BRError2DEvaluator::MakeShared(landmark, meas);
  EXPECT_MATRIX_NEAR(err_func->value(), Eigen::Vector2d::Zero(), tol);
  EXPECT_MATRIX_NEAR(err_func->forward()->value(), Eigen::Vector2d::Zero(),
                     tol);
  // feed in wrong meas
  err_func =
      BRError2DEvaluator::MakeShared(landmark, Eigen::Vector2d(0.0, 0.0));
  EXPECT_MATRIX_NEAR(err_func->value(), meas, tol);
  EXPECT_MATRIX_NEAR(err_func->forward()->value(), meas, tol);
}

// Test Jacobians
TEST(BearingRange2DEval, Jacobian) {
  // landmark at x=1 y=1, NOTE: Z component is ignored
  auto lm_val = Eigen::Vector4d(1.0, 2.0, 1.0, 1.0);
  auto landmark = vspace::VSpaceStateVar<4>::MakeShared(lm_val);
  auto meas = Eigen::Vector2d(M_PI / 4, sqrt(2));
  // define error function
  const auto err_func = BRError2DEvaluator::MakeShared(landmark, meas);
  // get analytic jacobian
  auto lhs = Eigen::Matrix<double, 2, 2>::Identity();
  Jacobians jacs;
  const auto node = err_func->forward();
  err_func->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  Eigen::Matrix<double, 2, 4> ajac = jacmap.at(landmark->key());
  // get numerical jacobian
  double eps = 1e-5;
  auto njac = Eigen::Matrix<double, 2, 4>::Zero().eval();
  for (int i = 0; i < 4; i++) {
    // Define perturbation
    Eigen::Vector4d pert = Eigen::Vector4d::Zero();
    pert(i) = eps;
    // Evaluate and undo perturbation
    landmark->update(pert);
    Eigen::Vector2d err1 = err_func->value();
    landmark->update(-2 * pert);
    Eigen::Vector2d err2 = err_func->value();
    landmark->update(pert);
    // store numerical derivative
    njac.block<2, 1>(0, i) = (err1 - err2) / (2 * eps);
  }
  cout << "Analytic Jacobian" << endl << ajac << endl;
  cout << "Numerical Jacobian" << endl << njac << endl;
  EXPECT_MATRIX_NEAR(ajac, njac, 1e-5);
}

// Test Jacobians
TEST(BearingRange2DEval, JacobianRangeZero) {
  // landmark at x=1 y=1, NOTE: Z component is ignored
  auto lm_val = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
  auto landmark = vspace::VSpaceStateVar<4>::MakeShared(lm_val);
  auto meas = Eigen::Vector2d(0, 0);
  // define error function
  const auto err_func = BRError2DEvaluator::MakeShared(landmark, meas);
  // get analytic jacobian
  auto lhs = Eigen::Matrix<double, 2, 2>::Identity();
  Jacobians jacs;
  const auto node = err_func->forward();
  err_func->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  Eigen::Matrix<double, 2, 4> ajac = jacmap.at(landmark->key());
  // get expected jacobian
  auto njac = Eigen::Matrix<double, 2, 4>::Zero().eval();
  cout << "Analytic Jacobian" << endl << ajac << endl;
  cout << "Expected Jacobian" << endl << njac << endl;
  EXPECT_MATRIX_NEAR(ajac, njac, 1e-5);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}