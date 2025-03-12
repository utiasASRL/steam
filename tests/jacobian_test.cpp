#include <gtest/gtest.h>

#include <iostream>
#include <random>

#include "lgmath.hpp"
#include "steam.hpp"

#include "steam/trajectory/const_acc/acceleration_interpolator.hpp"
#include "steam/trajectory/const_acc/pose_interpolator.hpp"
#include "steam/trajectory/const_acc/velocity_interpolator.hpp"
#include "steam/trajectory/const_vel/pose_interpolator.hpp"
#include "steam/trajectory/const_vel/velocity_interpolator.hpp"
#include "steam/trajectory/singer/acceleration_interpolator.hpp"
#include "steam/trajectory/singer/pose_interpolator.hpp"
#include "steam/trajectory/singer/velocity_interpolator.hpp"
#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include "steam/problem/cost_term/preintegrated_imu_cost_term.hpp"
#include "steam/problem/cost_term/p2p_global_perturb_super_cost_term.hpp"

TEST(ConstAcc, PoseInterpolator) {
  using namespace steam::traj::const_acc;
  using namespace steam;
  // TODO: try a bunch of random dw_01_in1;
  const Eigen::Matrix<double, 6, 1> dw_01_in1 =
      -1 * Eigen::Matrix<double, 6, 1>::Ones();
  const Eigen::Matrix<double, 6, 1> dw_02_in2 = dw_01_in1;
  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Zero();
  lgmath::se3::Transformation T_10;

  const double dt = 0.1;

  Eigen::Matrix<double, 6, 1> w_02_in2 = dw_01_in1 * dt;

  // TODO: try different values of tau between 0 and dt
  steam::traj::Time t1(0.);
  steam::traj::Time t2(dt);
  steam::traj::Time tau(dt / 2);

  // Use Magnus expansion to extrapolate pose using constant acceleration
  lgmath::se3::Transformation T_20 =
      lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
          w_01_in1 * dt + 0.5 * dw_01_in1 * pow(dt, 2) +
          (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 3) +
          (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
              lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 5))) *
      T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto dw_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);
  const auto dw_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2);

  const auto knot1 =
      std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var);
  const auto knot2 =
      std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var);

  const auto T_q0_eval = PoseInterpolator::MakeShared(tau, knot1, knot2);

  // check the forward pass is what we expect
  {
    std::cout << T_q0_eval->evaluate().matrix() << std::endl;

    lgmath::se3::Transformation T_q0_expected =
        lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
            w_01_in1 * (dt / 2) + 0.5 * dw_01_in1 * pow((dt / 2), 2) +
            (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 *
                pow((dt / 2), 3) +
            (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
                lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 *
                pow((dt / 2), 5))) *
        T_10;
    EXPECT_LT((T_q0_expected.matrix() - T_q0_eval->evaluate().matrix()).norm(),
              1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = T_q0_eval->forward();
  T_q0_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var_mod1, w_01_in1_var, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var_mod2, w_01_in1_var, dw_01_in1_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod1, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod2, dw_01_in1_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "dw1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi1);
      const auto dw_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod1);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod2);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var_mod1, w_02_in2_var, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var_mod2, w_02_in2_var, dw_02_in2_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod1, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod2, dw_02_in2_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw2
  {
    std::cout << "dw2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi1);
      const auto dw_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod1);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod2);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(ConstAcc, VelocityInterpolator) {
  using namespace steam::traj::const_acc;
  using namespace steam;
  // TODO: try a bunch of random dw_01_in1;
  const Eigen::Matrix<double, 6, 1> dw_01_in1 =
      -1 * Eigen::Matrix<double, 6, 1>::Ones();
  const Eigen::Matrix<double, 6, 1> dw_02_in2 = dw_01_in1;
  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Zero();
  lgmath::se3::Transformation T_10;

  const double dt = 0.1;

  Eigen::Matrix<double, 6, 1> w_02_in2 = dw_01_in1 * dt;

  // TODO: try different values of tau between 0 and dt
  steam::traj::Time t1(0.);
  steam::traj::Time t2(dt);
  steam::traj::Time tau(dt / 2);

  // Use Magnus expansion to extrapolate pose using constant acceleration
  lgmath::se3::Transformation T_20 =
      lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
          w_01_in1 * dt + 0.5 * dw_01_in1 * pow(dt, 2) +
          (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 3) +
          (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
              lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 5))) *
      T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto dw_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);
  const auto dw_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2);

  const auto knot1 =
      std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var);
  const auto knot2 =
      std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var);

  const auto w_0q_inq_eval =
      VelocityInterpolator::MakeShared(tau, knot1, knot2);

  // check the forward pass is what we expect
  {
    std::cout << w_0q_inq_eval->evaluate().transpose() << std::endl;
    Eigen::Matrix<double, 6, 1> w_0q_inq_expected = dw_01_in1 * (dt / 2);
    // std::cout << dw_01_in1.transpose() << std::endl;
    EXPECT_LT((w_0q_inq_eval->evaluate() - w_0q_inq_expected).norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = w_0q_inq_eval->forward();
  w_0q_inq_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var_mod1, w_01_in1_var, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var_mod2, w_01_in1_var, dw_01_in1_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod1, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod2, dw_01_in1_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw1
  {
    std::cout << "dw1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi1);
      const auto dw_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod1);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod2);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var_mod1, w_02_in2_var, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var_mod2, w_02_in2_var, dw_02_in2_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod1, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod2, dw_02_in2_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw2
  {
    std::cout << "dw2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi1);
      const auto dw_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod1);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod2);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(ConstAcc, AccelerationInterpolator) {
  using namespace steam::traj::const_acc;
  using namespace steam;
  // TODO: try a bunch of random dw_01_in1;
  const Eigen::Matrix<double, 6, 1> dw_01_in1 =
      -1 * Eigen::Matrix<double, 6, 1>::Ones();
  const Eigen::Matrix<double, 6, 1> dw_02_in2 = dw_01_in1;
  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Zero();
  lgmath::se3::Transformation T_10;

  const double dt = 0.1;

  Eigen::Matrix<double, 6, 1> w_02_in2 = dw_01_in1 * dt;

  // TODO: try different values of tau between 0 and dt
  steam::traj::Time t1(0.);
  steam::traj::Time t2(dt);
  steam::traj::Time tau(dt / 2);

  // Use Magnus expansion to extrapolate pose using constant acceleration
  lgmath::se3::Transformation T_20 =
      lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
          w_01_in1 * dt + 0.5 * dw_01_in1 * pow(dt, 2) +
          (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 3) +
          (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
              lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 5))) *
      T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto dw_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);
  const auto dw_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2);

  const auto knot1 =
      std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var);
  const auto knot2 =
      std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var);

  const auto dw_0q_inq_eval =
      AccelerationInterpolator::MakeShared(tau, knot1, knot2);

  // check the forward pass is what we expect
  {
    std::cout << dw_0q_inq_eval->evaluate().transpose() << std::endl;
    std::cout << dw_01_in1.transpose() << std::endl;
    EXPECT_LT((dw_0q_inq_eval->evaluate() - dw_01_in1).norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = dw_0q_inq_eval->forward();
  dw_0q_inq_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var_mod1, w_01_in1_var, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var_mod2, w_01_in1_var, dw_01_in1_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod1, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod2, dw_01_in1_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw1
  {
    std::cout << "dw1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi1);
      const auto dw_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod1);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod2);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var_mod1, w_02_in2_var, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var_mod2, w_02_in2_var, dw_02_in2_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod1, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod2, dw_02_in2_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw2
  {
    std::cout << "dw2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi1);
      const auto dw_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod1);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod2);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(Singer, PoseInterpolator) {
  using namespace steam::traj::singer;
  using namespace steam;
  // TODO: try a bunch of random dw_01_in1;
  const Eigen::Matrix<double, 6, 1> dw_01_in1 =
      -1 * Eigen::Matrix<double, 6, 1>::Ones();
  const Eigen::Matrix<double, 6, 1> dw_02_in2 = dw_01_in1;
  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Zero();
  lgmath::se3::Transformation T_10;

  const double dt = 0.1;
  // TODO: try different values of tau between 0 and dt
  steam::traj::Time t1(0.);
  steam::traj::Time t2(dt);
  steam::traj::Time tau(dt / 2);

  Eigen::Matrix<double, 6, 1> w_02_in2 = dw_01_in1 * dt;

  // Use Magnus expansion to extrapolate pose using constant acceleration
  lgmath::se3::Transformation T_20 =
      lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
          w_01_in1 * dt + 0.5 * dw_01_in1 * pow(dt, 2) +
          (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 3) +
          (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
              lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 5))) *
      T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto dw_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);
  const auto dw_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2);

  const auto knot1 =
      std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var);
  const auto knot2 =
      std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var);

  Eigen::Matrix<double, 6, 1> ad = Eigen::Matrix<double, 6, 1>::Ones();
  const auto T_q0_eval = PoseInterpolator::MakeShared(tau, knot1, knot2, ad);

  // check the forward pass is what we expect
  {
    Eigen::Matrix<double, 6, 1> ad_ = Eigen::Matrix<double, 6, 1>::Zero();
    const auto T_q0_eval_ =
        PoseInterpolator::MakeShared(tau, knot1, knot2, ad_);
    std::cout << T_q0_eval_->evaluate().matrix() << std::endl;

    lgmath::se3::Transformation T_q0_expected =
        lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
            w_01_in1 * (dt / 2) + 0.5 * dw_01_in1 * pow((dt / 2), 2) +
            (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 *
                pow((dt / 2), 3) +
            (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
                lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 *
                pow((dt / 2), 5))) *
        T_10;
    EXPECT_LT((T_q0_expected.matrix() - T_q0_eval_->evaluate().matrix()).norm(),
              1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = T_q0_eval->forward();
  T_q0_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var_mod1, w_01_in1_var, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var_mod2, w_01_in1_var, dw_01_in1_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod1, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod2, dw_01_in1_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "dw1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi1);
      const auto dw_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod1);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod2);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var_mod1, w_02_in2_var, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var_mod2, w_02_in2_var, dw_02_in2_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod1, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod2, dw_02_in2_var);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw2
  {
    std::cout << "dw2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi1);
      const auto dw_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod1);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod2);

      const auto T_q0_eval_mod1 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto T_q0_eval_mod2 =
          PoseInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (T_q0_eval_mod1->evaluate() * T_q0_eval_mod2->evaluate().inverse())
              .vec() /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(Singer, VelocityInterpolator) {
  using namespace steam::traj::singer;
  using namespace steam;
  // TODO: try a bunch of random dw_01_in1;
  const Eigen::Matrix<double, 6, 1> dw_01_in1 =
      -1 * Eigen::Matrix<double, 6, 1>::Ones();
  const Eigen::Matrix<double, 6, 1> dw_02_in2 = dw_01_in1;
  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Zero();
  lgmath::se3::Transformation T_10;

  const double dt = 0.1;

  Eigen::Matrix<double, 6, 1> w_02_in2 = dw_01_in1 * dt;

  // TODO: try different values of tau between 0 and dt
  steam::traj::Time t1(0.);
  steam::traj::Time t2(dt);
  steam::traj::Time tau(dt / 2);

  // Use Magnus expansion to extrapolate pose using constant acceleration
  lgmath::se3::Transformation T_20 =
      lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
          w_01_in1 * dt + 0.5 * dw_01_in1 * pow(dt, 2) +
          (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 3) +
          (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
              lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 5))) *
      T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto dw_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);
  const auto dw_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2);

  const auto knot1 =
      std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var);
  const auto knot2 =
      std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var);

  Eigen::Matrix<double, 6, 1> ad = Eigen::Matrix<double, 6, 1>::Ones();

  const auto w_0q_inq_eval =
      VelocityInterpolator::MakeShared(tau, knot1, knot2, ad);

  // check the forward pass is what we expect
  {
    Eigen::Matrix<double, 6, 1> ad_ = Eigen::Matrix<double, 6, 1>::Zero();
    const auto w_0q_inq_eval_ =
        VelocityInterpolator::MakeShared(tau, knot1, knot2, ad_);
    std::cout << w_0q_inq_eval_->evaluate().transpose() << std::endl;
    Eigen::Matrix<double, 6, 1> w_0q_inq_expected = dw_01_in1 * (dt / 2);
    // std::cout << dw_01_in1.transpose() << std::endl;
    EXPECT_LT((w_0q_inq_eval_->evaluate() - w_0q_inq_expected).norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = w_0q_inq_eval->forward();
  w_0q_inq_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var_mod1, w_01_in1_var, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var_mod2, w_01_in1_var, dw_01_in1_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod1, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod2, dw_01_in1_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw1
  {
    std::cout << "dw1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi1);
      const auto dw_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod1);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod2);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var_mod1, w_02_in2_var, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var_mod2, w_02_in2_var, dw_02_in2_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod1, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod2, dw_02_in2_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw2
  {
    std::cout << "dw2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi1);
      const auto dw_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod1);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod2);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(Singer, AccelerationInterpolator) {
  using namespace steam::traj::singer;
  using namespace steam;
  // TODO: try a bunch of random dw_01_in1;
  const Eigen::Matrix<double, 6, 1> dw_01_in1 =
      -1 * Eigen::Matrix<double, 6, 1>::Ones();
  const Eigen::Matrix<double, 6, 1> dw_02_in2 = dw_01_in1;
  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Zero();
  lgmath::se3::Transformation T_10;

  const double dt = 0.1;

  Eigen::Matrix<double, 6, 1> w_02_in2 = dw_01_in1 * dt;

  // TODO: try different values of tau between 0 and dt
  steam::traj::Time t1(0.);
  steam::traj::Time t2(dt);
  steam::traj::Time tau(dt / 2);

  // Use Magnus expansion to extrapolate pose using constant acceleration
  lgmath::se3::Transformation T_20 =
      lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
          w_01_in1 * dt + 0.5 * dw_01_in1 * pow(dt, 2) +
          (1 / 12) * lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 3) +
          (1 / 240) * lgmath::se3::curlyhat(dw_01_in1) *
              lgmath::se3::curlyhat(dw_01_in1) * w_01_in1 * pow(dt, 5))) *
      T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto dw_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);
  const auto dw_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2);

  const auto knot1 =
      std::make_shared<Variable>(t1, T_10_var, w_01_in1_var, dw_01_in1_var);
  const auto knot2 =
      std::make_shared<Variable>(t2, T_20_var, w_02_in2_var, dw_02_in2_var);

  Eigen::Matrix<double, 6, 1> ad = Eigen::Matrix<double, 6, 1>::Ones();

  const auto dw_0q_inq_eval =
      AccelerationInterpolator::MakeShared(tau, knot1, knot2, ad);

  // check the forward pass is what we expect
  {
    Eigen::Matrix<double, 6, 1> ad_ = Eigen::Matrix<double, 6, 1>::Zero();
    const auto dw_0q_inq_eval_ =
        AccelerationInterpolator::MakeShared(tau, knot1, knot2, ad_);
    std::cout << dw_0q_inq_eval_->evaluate().transpose() << std::endl;
    std::cout << dw_01_in1.transpose() << std::endl;
    EXPECT_LT((dw_0q_inq_eval_->evaluate() - dw_01_in1).norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = dw_0q_inq_eval->forward();
  dw_0q_inq_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var_mod1, w_01_in1_var, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var_mod2, w_01_in1_var, dw_01_in1_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod1, dw_01_in1_var);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var_mod2, dw_01_in1_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw1
  {
    std::cout << "dw1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi1);
      const auto dw_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_01_in1 + xi2);
      const auto knot1_mod1 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod1);
      const auto knot1_mod2 = std::make_shared<Variable>(
          t1, T_10_var, w_01_in1_var, dw_01_in1_var_mod2);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod1, knot2, ad);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1_mod2, knot2, ad);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var_mod1, w_02_in2_var, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var_mod2, w_02_in2_var, dw_02_in2_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod1, dw_02_in2_var);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var_mod2, dw_02_in2_var);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw2
  {
    std::cout << "dw2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(dw_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi1);
      const auto dw_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(dw_02_in2 + xi2);
      const auto knot2_mod1 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod1);
      const auto knot2_mod2 = std::make_shared<Variable>(
          t2, T_20_var, w_02_in2_var, dw_02_in2_var_mod2);

      const auto dw_0q_inq_eval_mod1 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod1, ad);
      const auto dw_0q_inq_eval_mod2 =
          AccelerationInterpolator::MakeShared(tau, knot1, knot2_mod2, ad);
      njac.block<6, 1>(0, j) =
          (dw_0q_inq_eval_mod1->evaluate() - dw_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(IMU, AccelerationErrorEvaluator) {
  using namespace steam::imu;
  using namespace steam;

  Eigen::Matrix<double, 6, 1> xi_vm;
  xi_vm << 1, 2, 3, 0.2, 0.3, 0.6;
  lgmath::se3::Transformation T_vm(xi_vm);

  Eigen::Matrix<double, 6, 1> xi_mi;
  xi_mi << 0, 0, 0, 0.2, 0.2, 0;
  lgmath::se3::Transformation T_mi(xi_mi);

  Eigen::Matrix<double, 6, 1> dw;
  dw << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix<double, 6, 1> bias = 0.05 * Eigen::Matrix<double, 6, 1>::Ones();

  Eigen::Matrix<double, 3, 1> g;
  g << 0, 0, -9.8042;

  const Eigen::Matrix<double, 3, 1> acc_meas =
      -dw.block<3, 1>(0, 0) -
      T_vm.matrix().block<3, 3>(0, 0) * T_mi.matrix().block<3, 3>(0, 0) * g +
      bias.block<3, 1>(0, 0);

  const auto T_vm_var = se3::SE3StateVar::MakeShared(T_vm);
  const auto T_mi_var = se3::SE3StateVar::MakeShared(T_mi);
  const auto dw_var = vspace::VSpaceStateVar<6>::MakeShared(dw);
  const auto bias_var = vspace::VSpaceStateVar<6>::MakeShared(bias);

  const auto acc_err_eval =
      AccelerationError(T_vm_var, dw_var, bias_var, T_mi_var, acc_meas);

  // check the forward pass is what we expect
  {
    std::cout << acc_err_eval->evaluate() << std::endl;
    EXPECT_LT(acc_err_eval->evaluate().norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 3, 3> lhs = Eigen::Matrix<double, 3, 3>::Identity();
  const auto node = acc_err_eval->forward();
  acc_err_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();

  // T_vm
  {
    std::cout << "T_vm:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(T_vm_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_vm_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_vm);
      const auto T_vm_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_vm);
      const auto acc_err_eval_mod1 = AccelerationError(
          T_vm_var_mod1, dw_var, bias_var, T_mi_var, acc_meas);
      const auto acc_err_eval_mod2 = AccelerationError(
          T_vm_var_mod2, dw_var, bias_var, T_mi_var, acc_meas);

      njac.block<3, 1>(0, j) =
          (acc_err_eval_mod1->evaluate() - acc_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T_mi
  {
    std::cout << "T_mi:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(T_mi_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_mi_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_mi);
      const auto T_mi_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_mi);
      const auto acc_err_eval_mod1 = AccelerationError(
          T_vm_var, dw_var, bias_var, T_mi_var_mod1, acc_meas);
      const auto acc_err_eval_mod2 = AccelerationError(
          T_vm_var, dw_var, bias_var, T_mi_var_mod2, acc_meas);

      njac.block<3, 1>(0, j) =
          (acc_err_eval_mod1->evaluate() - acc_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // dw
  {
    std::cout << "dw:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(dw_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(dw + xi1);
      const auto dw_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(dw + xi2);
      const auto acc_err_eval_mod1 = AccelerationError(
          T_vm_var, dw_var_mod1, bias_var, T_mi_var, acc_meas);
      const auto acc_err_eval_mod2 = AccelerationError(
          T_vm_var, dw_var_mod2, bias_var, T_mi_var, acc_meas);

      njac.block<3, 1>(0, j) =
          (acc_err_eval_mod1->evaluate() - acc_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // bias
  {
    std::cout << "bias:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(bias_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto bias_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(bias + xi1);
      const auto bias_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(bias + xi2);
      const auto acc_err_eval_mod1 = AccelerationError(
          T_vm_var, dw_var, bias_var_mod1, T_mi_var, acc_meas);
      const auto acc_err_eval_mod2 = AccelerationError(
          T_vm_var, dw_var, bias_var_mod2, T_mi_var, acc_meas);

      njac.block<3, 1>(0, j) =
          (acc_err_eval_mod1->evaluate() - acc_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(IMU, GyroErrorEvaluator) {
  using namespace steam::imu;
  using namespace steam;

  Eigen::Matrix<double, 6, 1> dw;
  dw << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix<double, 6, 1> bias = 0.05 * Eigen::Matrix<double, 6, 1>::Ones();

  const Eigen::Matrix<double, 3, 1> gyro_meas =
      -dw.block<3, 1>(3, 0) + bias.block<3, 1>(3, 0);

  const auto dw_var = vspace::VSpaceStateVar<6>::MakeShared(dw);
  const auto bias_var = vspace::VSpaceStateVar<6>::MakeShared(bias);

  const auto gyro_err_eval = GyroError(dw_var, bias_var, gyro_meas);

  // check the forward pass is what we expect
  {
    std::cout << gyro_err_eval->evaluate() << std::endl;
    EXPECT_LT(gyro_err_eval->evaluate().norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 3, 3> lhs = Eigen::Matrix<double, 3, 3>::Identity();
  const auto node = gyro_err_eval->forward();
  gyro_err_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();

  // dw
  {
    std::cout << "dw:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(dw_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto dw_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(dw + xi1);
      const auto dw_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(dw + xi2);
      const auto gyro_err_eval_mod1 =
          GyroError(dw_var_mod1, bias_var, gyro_meas);
      const auto gyro_err_eval_mod2 =
          GyroError(dw_var_mod2, bias_var, gyro_meas);

      njac.block<3, 1>(0, j) =
          (gyro_err_eval_mod1->evaluate() - gyro_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // bias
  {
    std::cout << "bias:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(bias_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto bias_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(bias + xi1);
      const auto bias_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(bias + xi2);
      const auto gyro_err_eval_mod1 =
          GyroError(dw_var, bias_var_mod1, gyro_meas);
      const auto gyro_err_eval_mod2 =
          GyroError(dw_var, bias_var_mod2, gyro_meas);

      njac.block<3, 1>(0, j) =
          (gyro_err_eval_mod1->evaluate() - gyro_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(P2P, P2PErrorDopplerEvaluator) {
  using namespace steam;
  using namespace steam::p2p;
  Eigen::Matrix<double, 6, 1> w_m_v_in_v;
  w_m_v_in_v << 1, 2, 3, 0.1, 0.2, 0.3;
  Eigen::Matrix<double, 6, 1> xi_vm, xi_vs;
  xi_vm << 4, 5, 6, 0.3, 0.2, 0.3;
  xi_vs << 0.1, 0.2, 0.3, 0.03, 0.02, 0.01;
  const auto T_vm = lgmath::se3::Transformation(xi_vm);
  const Eigen::Matrix4d T_mv = T_vm.inverse().matrix();
  const Eigen::Matrix4d T_vs = lgmath::se3::Transformation(xi_vs).matrix();
  const Eigen::Matrix4d T_ms = T_mv * T_vs;

  Eigen::Vector3d query;
  query << 10, 20, 30;
  const Eigen::Vector3d abar = query.normalized();
  const double beta = 0.0535;
  const bool rm_ori = false;
  const Eigen::Vector3d delta_q =
      beta * abar * abar.transpose() *
      lgmath::se3::point2fs(query).block<3, 6>(0, 0) *
      lgmath::se3::tranAd(T_vs.inverse()) * w_m_v_in_v;
  const Eigen::Vector3d reference =
      T_ms.block<3, 3>(0, 0) * (query + delta_q) + T_ms.block<3, 1>(0, 3);

  const auto T_vm_var = se3::SE3StateVar::MakeShared(T_vm);
  const auto w_mv_in_v_var = vspace::VSpaceStateVar<6>::MakeShared(w_m_v_in_v);

  const auto p2p_err_eval =
      p2pErrorDoppler(T_vm_var, w_mv_in_v_var, reference, query, beta, rm_ori);

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 3, 3> lhs = Eigen::Matrix<double, 3, 3>::Identity();
  const auto node = p2p_err_eval->forward();
  p2p_err_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();

  // T_vm
  {
    std::cout << "T_vm:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(T_vm_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_vm_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_vm);
      const auto T_vm_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_vm);

      const auto p2p_err_eval_mod1 =
          p2pErrorDoppler(T_vm_var_mod1, w_mv_in_v_var, reference, query, beta, rm_ori);
      const auto p2p_err_eval_mod2 =
          p2pErrorDoppler(T_vm_var_mod2, w_mv_in_v_var, reference, query, beta, rm_ori);

      njac.block<3, 1>(0, j) =
          (p2p_err_eval_mod1->evaluate() - p2p_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w_m_v_in_v
  {
    std::cout << "w_mv_in_v:" << std::endl;
    Eigen::Matrix<double, 3, 6> ajac = jacmap.at(w_mv_in_v_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 3, 6> njac = Eigen::Matrix<double, 3, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_mv_in_v_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_m_v_in_v + xi1);
      const auto w_mv_in_v_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_m_v_in_v + xi2);

      const auto p2p_err_eval_mod1 =
          p2pErrorDoppler(T_vm_var, w_mv_in_v_var_mod1, reference, query, beta, rm_ori);
      const auto p2p_err_eval_mod2 =
          p2pErrorDoppler(T_vm_var, w_mv_in_v_var_mod2, reference, query, beta, rm_ori);

      njac.block<3, 1>(0, j) =
          (p2p_err_eval_mod1->evaluate() - p2p_err_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(ConstVel, VelocityInterpolator) {
  using namespace steam::traj::const_vel;
  using namespace steam;

  Eigen::Matrix<double, 6, 1> w_01_in1 = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<double, 6, 1> w_02_in2 = Eigen::Matrix<double, 6, 1>::Ones();
  lgmath::se3::Transformation T_10;

  const double dt = 0.1;
  const auto accel = (w_02_in2 - w_01_in1) / dt;

  // TODO: try different values of tau between 0 and dt
  steam::traj::Time t1(0.);
  steam::traj::Time t2(dt);
  steam::traj::Time tau(dt / 2);

  lgmath::se3::Transformation T_20 =
      lgmath::se3::Transformation(Eigen::Matrix<double, 6, 1>(
          w_01_in1 * dt + 0.5 * accel * pow(dt, 2) +
          (1 / 12) * lgmath::se3::curlyhat(accel) * w_01_in1 * pow(dt, 3) +
          (1 / 240) * lgmath::se3::curlyhat(accel) *
              lgmath::se3::curlyhat(accel) * w_01_in1 * pow(dt, 5))) *
      T_10;

  const auto T_10_var = se3::SE3StateVar::MakeShared(T_10);
  const auto w_01_in1_var = vspace::VSpaceStateVar<6>::MakeShared(w_01_in1);
  const auto T_20_var = se3::SE3StateVar::MakeShared(T_20);
  const auto w_02_in2_var = vspace::VSpaceStateVar<6>::MakeShared(w_02_in2);

  const auto knot1 = std::make_shared<Variable>(t1, T_10_var, w_01_in1_var);
  const auto knot2 = std::make_shared<Variable>(t2, T_20_var, w_02_in2_var);

  const auto w_0q_inq_eval =
      VelocityInterpolator::MakeShared(tau, knot1, knot2);

  // check the forward pass is what we expect
  {
    std::cout << w_0q_inq_eval->evaluate().transpose() << std::endl;
    Eigen::Matrix<double, 6, 1> w_0q_inq_expected = w_01_in1 + accel * dt / 2;
    // std::cout << dw_01_in1.transpose() << std::endl;
    EXPECT_LT((w_0q_inq_eval->evaluate() - w_0q_inq_expected).norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 6, 6> lhs = Eigen::Matrix<double, 6, 6>::Identity();
  const auto node = w_0q_inq_eval->forward();
  w_0q_inq_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();
  // T1
  {
    std::cout << "T1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_10_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_10_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_10);
      const auto T_10_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_10);
      const auto knot1_mod1 =
          std::make_shared<Variable>(t1, T_10_var_mod1, w_01_in1_var);
      const auto knot1_mod2 =
          std::make_shared<Variable>(t1, T_10_var_mod2, w_01_in1_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w1
  {
    std::cout << "w1:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_01_in1_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_01_in1_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi1);
      const auto w_01_in1_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_01_in1 + xi2);
      const auto knot1_mod1 =
          std::make_shared<Variable>(t1, T_10_var, w_01_in1_var_mod1);
      const auto knot1_mod2 =
          std::make_shared<Variable>(t1, T_10_var, w_01_in1_var_mod2);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1_mod1, knot2);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1_mod2, knot2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // T2
  {
    std::cout << "T2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(T_20_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto T_20_var_mod1 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi1) * T_20);
      const auto T_20_var_mod2 =
          se3::SE3StateVar::MakeShared(lgmath::se3::Transformation(xi2) * T_20);
      const auto knot2_mod1 =
          std::make_shared<Variable>(t2, T_20_var_mod1, w_02_in2_var);
      const auto knot2_mod2 =
          std::make_shared<Variable>(t2, T_20_var_mod2, w_02_in2_var);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    // EXPECT_LT((njac - ajac).norm(), 1e-2);
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }

  // w2
  {
    std::cout << "w2:" << std::endl;
    Eigen::Matrix<double, 6, 6> ajac = jacmap.at(w_02_in2_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 6, 6> njac = Eigen::Matrix<double, 6, 6>::Zero();
    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      const auto w_02_in2_var_mod1 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi1);
      const auto w_02_in2_var_mod2 =
          vspace::VSpaceStateVar<6>::MakeShared(w_02_in2 + xi2);
      const auto knot2_mod1 =
          std::make_shared<Variable>(t2, T_20_var, w_02_in2_var_mod1);
      const auto knot2_mod2 =
          std::make_shared<Variable>(t2, T_20_var, w_02_in2_var_mod2);

      const auto w_0q_inq_eval_mod1 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod1);
      const auto w_0q_inq_eval_mod2 =
          VelocityInterpolator::MakeShared(tau, knot1, knot2_mod2);
      njac.block<6, 1>(0, j) =
          (w_0q_inq_eval_mod1->evaluate() - w_0q_inq_eval_mod2->evaluate()) /
          (2 * eps);
    }
    std::cout << njac << std::endl;
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(IMU, PreintIMUCostTerm) {
  using namespace steam::imu;
  using namespace steam;
  using namespace steam::traj;

  Eigen::Vector3d phi = {0, 0, 0.1};
  Eigen::Matrix3d C_ij = lgmath::so3::vec2rot(phi);
  Eigen::Vector3d r_ij = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_ij = Eigen::Vector3d::Zero();

  Eigen::Vector3d lin_acc = {1.0, 0.0, 9.8042};
  Eigen::Vector3d gravity = {0.0, 0.0, -9.8042};
  Eigen::Vector3d ang_vel = {0.0, 0.0, 1.0};
  const double delta_t = 0.01;
  std::vector<IMUData> imu_data_vec;
  for (int i = 1; i < 10; ++i) {
    imu_data_vec.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
    r_ij += v_ij * delta_t + 0.5 * ((C_ij * lin_acc) + gravity) * delta_t * delta_t;
    v_ij += ((C_ij * lin_acc) + gravity) * delta_t;
    C_ij = C_ij * lgmath::so3::vec2rot(delta_t * ang_vel);
  }
  r_ij += v_ij * delta_t + 0.5 * ((C_ij * lin_acc) + gravity) * delta_t * delta_t;
  v_ij += ((C_ij * lin_acc) + gravity) * delta_t;
  C_ij = C_ij * lgmath::so3::vec2rot(delta_t * ang_vel);

  Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
  T_i.block<3, 3>(0, 0) = lgmath::so3::vec2rot(phi);
  auto transform_i = lgmath::se3::Transformation(T_i);
  Eigen::Vector3d v_i = Eigen::Vector3d::Zero();

  Eigen::Matrix3d C_j = C_ij;
  Eigen::Vector3d r_j = r_ij;
  Eigen::Matrix4d T_j = Eigen::Matrix4d::Identity();
  T_j.block<3, 3>(0, 0) = C_j;
  T_j.block<3, 1>(0, 3) = r_j;
  auto transform_j = lgmath::se3::Transformation(T_j);
  Eigen::Vector3d v_j = v_ij;

  Eigen::Matrix<double, 6, 1> bias = Eigen::Matrix<double, 6, 1>::Zero();

  const auto T_i_var = se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_i));
  const auto v_i_var = vspace::PreIntVelocityStateVar<3>::MakeShared(v_i, T_i_var);
  const auto T_j_var = se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_j));
  const auto v_j_var = vspace::PreIntVelocityStateVar<3>::MakeShared(v_j, T_j_var);
  const auto bias_var = vspace::VSpaceStateVar<6>::MakeShared(bias);

  auto options = PreintIMUCostTerm::Options();
  options.gravity = gravity;
  options.loss_func = PreintIMUCostTerm::LOSS_FUNC::CAUCHY;

  const auto preint_cost_term = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var, v_j_var, bias_var, options);
  preint_cost_term->set(imu_data_vec);

  // check the forward pass is what we expect
  {
    const auto error = preint_cost_term->get_error();
    std::cout << error << std::endl;
    EXPECT_LT(error.norm(), 1e-6);
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Eigen::Matrix<double, 9, 24> jac = preint_cost_term->get_jacobian();

  // T_i
  {
    std::cout << "T_i:" << std::endl;
    Eigen::Matrix<double, 9, 6> ajac = jac.block<9, 6>(0, 0);
    std::cout << ajac << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    Eigen::Matrix<double, 9, 6> njac = Eigen::Matrix<double, 9, 6>::Zero();

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Matrix4d T_1 = T_i;
      T_1.block<3, 1>(0, 3) += T_i.block<3, 3>(0, 0) * xi1;
      Eigen::Matrix4d T_2 = T_i;
      T_2.block<3, 1>(0, 3) += T_i.block<3, 3>(0, 0) * xi2;
      const auto T_i_var_mod1 =
          se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_1));
      const auto T_i_var_mod2 =
          se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_2));

      const auto preint_cost_term_mod1 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var_mod1, T_j_var, v_i_var, v_j_var, bias_var, options);
      const auto preint_cost_term_mod2 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var_mod2, T_j_var, v_i_var, v_j_var, bias_var, options);
      std::vector<IMUData> imu_data_vec1;
      std::vector<IMUData> imu_data_vec2;
      for (int i = 1; i < 10; ++i) {
        imu_data_vec1.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
        imu_data_vec2.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
      }
      preint_cost_term_mod1->set(imu_data_vec1);
      preint_cost_term_mod2->set(imu_data_vec2);

      njac.block<9, 1>(0, j) =
          (preint_cost_term_mod1->get_error() - preint_cost_term_mod2->get_error()) /
          (2 * eps);
    }

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Matrix4d T_1 = Eigen::Matrix4d::Identity();
      T_1.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi1);
      Eigen::Matrix4d T_2 = Eigen::Matrix4d::Identity();
      T_2.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi2);
      const auto T_i_var_mod1 =
          se3::SE3StateVarGlobalPerturb::MakeShared(T_i * lgmath::se3::Transformation(T_1));
      const auto T_i_var_mod2 =
          se3::SE3StateVarGlobalPerturb::MakeShared(T_i * lgmath::se3::Transformation(T_2));


      const auto preint_cost_term_mod1 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var_mod1, T_j_var, v_i_var, v_j_var, bias_var, options);
      const auto preint_cost_term_mod2 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var_mod2, T_j_var, v_i_var, v_j_var, bias_var, options);
      preint_cost_term_mod1->set(imu_data_vec);
      preint_cost_term_mod2->set(imu_data_vec);

      njac.block<9, 1>(0, 3 + j) =
          (preint_cost_term_mod1->get_error() - preint_cost_term_mod2->get_error()) /
          (2 * eps);
    }

    std::cout << njac << std::endl;
    for (int i = 0; i < njac.rows(); ++i) {
      for (int j = 0; j < njac.cols(); ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
      }
    }
  }

  // T_j
  {
    std::cout << "T_j:" << std::endl;
    Eigen::Matrix<double, 9, 6> ajac = jac.block<9, 6>(0, 9);
    std::cout << ajac << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    Eigen::Matrix<double, 9, 6> njac = Eigen::Matrix<double, 9, 6>::Zero();

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Matrix4d T_1 = T_j;
      T_1.block<3, 1>(0, 3) += T_j.block<3, 3>(0, 0) * xi1;
      Eigen::Matrix4d T_2 = T_j;
      T_2.block<3, 1>(0, 3) += T_j.block<3, 3>(0, 0) * xi2;
      const auto T_j_var_mod1 =
          se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_1));
      const auto T_j_var_mod2 =
          se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_2));

      const auto preint_cost_term_mod1 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var_mod1, v_i_var, v_j_var, bias_var, options);
      const auto preint_cost_term_mod2 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var_mod2, v_i_var, v_j_var, bias_var, options);
      std::vector<IMUData> imu_data_vec1;
      std::vector<IMUData> imu_data_vec2;
      for (int i = 1; i < 10; ++i) {
        imu_data_vec1.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
        imu_data_vec2.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
      }
      preint_cost_term_mod1->set(imu_data_vec1);
      preint_cost_term_mod2->set(imu_data_vec2);

      njac.block<9, 1>(0, j) =
          (preint_cost_term_mod1->get_error() - preint_cost_term_mod2->get_error()) /
          (2 * eps);
    }

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Matrix4d T_1 = Eigen::Matrix4d::Identity();
      T_1.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi1);
      Eigen::Matrix4d T_2 = Eigen::Matrix4d::Identity();
      T_2.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi2);
      const auto T_j_var_mod1 =
          se3::SE3StateVarGlobalPerturb::MakeShared(T_j * lgmath::se3::Transformation(T_1));
      const auto T_j_var_mod2 =
          se3::SE3StateVarGlobalPerturb::MakeShared(T_j * lgmath::se3::Transformation(T_2));


      const auto preint_cost_term_mod1 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var_mod1, v_i_var, v_j_var, bias_var, options);
      const auto preint_cost_term_mod2 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var_mod2, v_i_var, v_j_var, bias_var, options);
      std::vector<IMUData> imu_data_vec1;
      std::vector<IMUData> imu_data_vec2;
      for (int i = 1; i < 10; ++i) {
        imu_data_vec1.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
        imu_data_vec2.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
      }
      preint_cost_term_mod1->set(imu_data_vec1);
      preint_cost_term_mod2->set(imu_data_vec2);

      njac.block<9, 1>(0, 3 + j) =
          (preint_cost_term_mod1->get_error() - preint_cost_term_mod2->get_error()) /
          (2 * eps);
    }

    std::cout << njac << std::endl;
    for (int i = 0; i < njac.rows(); ++i) {
      for (int j = 0; j < njac.cols(); ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
      }
    }
  }

  // v_i
  {
    std::cout << "v_i:" << std::endl;
    Eigen::Matrix<double, 9, 3> ajac = jac.block<9, 3>(0, 6);
    std::cout << ajac << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    Eigen::Matrix<double, 9, 3> njac = Eigen::Matrix<double, 9, 3>::Zero();

    const Eigen::Matrix3d C_i = T_i_var->evaluate().matrix().block<3, 3>(0, 0);

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Vector3d v_1 = v_i + C_i * xi1;
      Eigen::Vector3d v_2 = v_i + C_i * xi2;
      
      const auto v_i_var_mod1 = vspace::PreIntVelocityStateVar<3>::MakeShared(v_1, T_i_var);
      const auto v_i_var_mod2 = vspace::PreIntVelocityStateVar<3>::MakeShared(v_2, T_i_var);

      const auto preint_cost_term_mod1 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var_mod1, v_j_var, bias_var, options);
      const auto preint_cost_term_mod2 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var_mod2, v_j_var, bias_var, options);
      preint_cost_term_mod1->set(imu_data_vec);
      preint_cost_term_mod2->set(imu_data_vec);

      njac.block<9, 1>(0, j) =
          (preint_cost_term_mod1->get_error() - preint_cost_term_mod2->get_error()) /
          (2 * eps);
    }

    std::cout << njac << std::endl;
    for (int i = 0; i < njac.rows(); ++i) {
      for (int j = 0; j < njac.cols(); ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
      }
    }
  }

  // v_j
  {
    std::cout << "v_j:" << std::endl;
    Eigen::Matrix<double, 9, 3> ajac = jac.block<9, 3>(0, 15);
    std::cout << ajac << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    Eigen::Matrix<double, 9, 3> njac = Eigen::Matrix<double, 9, 3>::Zero();

    const Eigen::Matrix3d C_j = T_j_var->evaluate().matrix().block<3, 3>(0, 0);

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Vector3d v_1 = v_j + C_j * xi1;
      Eigen::Vector3d v_2 = v_j + C_j * xi2;
      
      const auto v_j_var_mod1 = vspace::PreIntVelocityStateVar<3>::MakeShared(v_1, T_j_var);
      const auto v_j_var_mod2 = vspace::PreIntVelocityStateVar<3>::MakeShared(v_2, T_j_var);

      const auto preint_cost_term_mod1 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var, v_j_var_mod1, bias_var, options);
      const auto preint_cost_term_mod2 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var, v_j_var_mod2, bias_var, options);
      preint_cost_term_mod1->set(imu_data_vec);
      preint_cost_term_mod2->set(imu_data_vec);

      njac.block<9, 1>(0, j) =
          (preint_cost_term_mod1->get_error() - preint_cost_term_mod2->get_error()) /
          (2 * eps);
    }

    std::cout << njac << std::endl;
    for (int i = 0; i < njac.rows(); ++i) {
      for (int j = 0; j < njac.cols(); ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
      }
    }
  }

  // bias
  {
    std::cout << "bias:" << std::endl;
    Eigen::Matrix<double, 9, 6> ajac = jac.block<9, 6>(0, 18);
    std::cout << ajac << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    Eigen::Matrix<double, 9, 6> njac = Eigen::Matrix<double, 9, 6>::Zero();

    for (int j = 0; j < 6; ++j) {
      Eigen::Matrix<double, 6, 1> xi1 = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 1> xi2 = Eigen::Matrix<double, 6, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Matrix<double, 6, 1> b_1 = bias + xi1;
      Eigen::Matrix<double, 6, 1> b_2 = bias + xi2;
      
      const auto bias_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(b_1);
      const auto bias_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(b_2);

      const auto preint_cost_term_mod1 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var, v_j_var, bias_var_mod1, options);
      const auto preint_cost_term_mod2 = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var, v_j_var, bias_var_mod2, options);
      preint_cost_term_mod1->set(imu_data_vec);
      preint_cost_term_mod2->set(imu_data_vec);

      njac.block<9, 1>(0, j) =
          (preint_cost_term_mod1->get_error() - preint_cost_term_mod2->get_error()) /
          (2 * eps);
    }

    std::cout << njac << std::endl;
    for (int i = 0; i < njac.rows(); ++i) {
      for (int j = 0; j < njac.cols(); ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
      }
    }
  }

  // Test Covariance:
  {
    std::default_random_engine generator;
    const double std_dev = 0.25;
    const double meas_cov = std_dev * std_dev;
    std::normal_distribution<double> distribution(0.0, std_dev);
    options.r_imu_acc << meas_cov, meas_cov, meas_cov;
    options.r_imu_ang << meas_cov, meas_cov, meas_cov;

    // Do Monte Carlo simulation to get covariance of the error
    const int N = 10000;
    Eigen::Matrix<double, 9, 9> preint_cov = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> Sigma = Eigen::Matrix<double, 9, 9>::Zero();
    for (int i = 0; i < N; ++i) {
      imu_data_vec.clear();
      // simulate IMU noise
      for (int j = 0; j < 10; ++j) {
        Eigen::Vector3d ang_vel_noisy = ang_vel;
        Eigen::Vector3d lin_acc_noisy = lin_acc;
        ang_vel_noisy[0] += distribution(generator);
        ang_vel_noisy[1] += distribution(generator);
        ang_vel_noisy[2] += distribution(generator);
        lin_acc_noisy[0] += distribution(generator);
        lin_acc_noisy[1] += distribution(generator);
        lin_acc_noisy[2] += distribution(generator);
        imu_data_vec.push_back(IMUData(j * delta_t, ang_vel_noisy, lin_acc_noisy));
      }

      const auto preint_cost_term = PreintIMUCostTerm::MakeShared(Time(0.0), Time(0.1), T_i_var, T_j_var, v_i_var, v_j_var, bias_var, options);
      preint_cost_term->set(imu_data_vec);
      const Eigen::Matrix<double, 9, 1> error = preint_cost_term->get_error();
      if (i == 0) {
        const PreintegratedMeasurement preint_meas = preint_cost_term->preintegrate_();
        preint_cov = preint_meas.cov;
      }
      Sigma += error * error.transpose();
    }
    Sigma /= N;
    std::cout << preint_cov << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    std::cout << Sigma << std::endl;
    for (int i = 0; i < preint_cov.rows(); ++i) {
      for (int j = 0; j < preint_cov.cols(); ++j) {
        EXPECT_LT(fabs((Sigma(i, j) - preint_cov(i, j)) / preint_cov(i, j)), 0.01);
      }
    }
  }
  // Note: the classical preintegration makes some simplifying assumptions that makes their covariance not
  // match the numerical covariance when doing a unit test, still seems to be implemented correctly.
}

TEST(P2P, P2PGlobal) {
  using namespace steam;
  using namespace steam::p2p;
  
  Eigen::Matrix<double, 6, 1> xi;
  xi << 1, 2, 3, 0.5, 0.9, 1.2;
  const auto T_iv = lgmath::se3::Transformation(xi);

  Eigen::Vector3d query;
  query << 10, 20, 30;
  Eigen::Matrix4d T_iv_mat = T_iv.matrix();
  Eigen::Vector3d reference = T_iv_mat.block<3, 3>(0, 0) * query + T_iv_mat.block<3, 1>(0, 3);
  const Eigen::Vector3d normal = {1.0, 2.0, 3.0};

  const auto T_iv_var = se3::SE3StateVarGlobalPerturb::MakeShared(T_iv);
  const auto p2p_err_eval = p2planeGlobalError(T_iv_var, reference, query, normal);

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;
  Jacobians jacs;
  Eigen::Matrix<double, 3, 3> lhs = Eigen::Matrix<double, 3, 3>::Identity();
  const auto node = p2p_err_eval->forward();
  p2p_err_eval->backward(lhs, node, jacs);
  const auto& jacmap = jacs.get();

  // T_iv
  {
    std::cout << "T_iv:" << std::endl;
    Eigen::Matrix<double, 1, 6> ajac = jacmap.at(T_iv_var->key());
    std::cout << ajac << std::endl;
    Eigen::Matrix<double, 1, 6> njac = Eigen::Matrix<double, 1, 6>::Zero();

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Matrix4d T_1 = T_iv_mat;
      T_1.block<3, 1>(0, 3) += T_iv_mat.block<3, 3>(0, 0) * xi1;
      Eigen::Matrix4d T_2 = T_iv_mat;
      T_2.block<3, 1>(0, 3) += T_iv_mat.block<3, 3>(0, 0) * xi2;
      const auto T_iv_var_mod1 =
          se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_1));
      const auto T_iv_var_mod2 =
          se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_2));

      const auto p2p_err_eval_mod1 = p2planeGlobalError(T_iv_var_mod1, reference, query, normal);
      const auto p2p_err_eval_mod2 = p2planeGlobalError(T_iv_var_mod2, reference, query, normal);

      njac.block<1, 1>(0, j) =
          (p2p_err_eval_mod1->evaluate() - p2p_err_eval_mod2->evaluate()) /
          (2 * eps);
    }

    for (int j = 0; j < 3; ++j) {
      Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
      Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
      xi1(j, 0) = eps;
      xi2(j, 0) = -eps;
      Eigen::Matrix4d T_1 = Eigen::Matrix4d::Identity();
      T_1.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi1);
      Eigen::Matrix4d T_2 = Eigen::Matrix4d::Identity();
      T_2.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi2);
      const auto T_iv_var_mod1 =
          se3::SE3StateVarGlobalPerturb::MakeShared(T_iv * lgmath::se3::Transformation(T_1));
      const auto T_iv_var_mod2 =
          se3::SE3StateVarGlobalPerturb::MakeShared(T_iv * lgmath::se3::Transformation(T_2));


      const auto p2p_err_eval_mod1 = p2planeGlobalError(T_iv_var_mod1, reference, query, normal);
      const auto p2p_err_eval_mod2 = p2planeGlobalError(T_iv_var_mod2, reference, query, normal);

      njac.block<1, 1>(0, 3 + j) =
          (p2p_err_eval_mod1->evaluate() - p2p_err_eval_mod2->evaluate()) /
          (2 * eps);
    }

    std::cout << njac << std::endl;
    for (int i = 0; i < njac.rows(); ++i) {
      for (int j = 0; j < njac.cols(); ++j) {
        if (fabs(ajac(i, j)) > 1.0e-2)
          EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.02);
        else
          EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-2);
      }
    }
  }
}

TEST(P2P, P2PGlobalSuper) {
  using namespace steam::imu;
  using namespace steam;
  using namespace steam::traj;

  Eigen::Vector3d phi_i = {0, 0, 0.1};
  const Eigen::Matrix3d C_0 = lgmath::so3::vec2rot(phi_i);
  const Eigen::Vector3d r_0 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d v_0 = {1.0, 2.0, 3.0};
  

  Eigen::Vector3d lin_acc = {1.0, 0.0, 9.8042};
  Eigen::Vector3d gravity = {0.0, 0.0, -9.8042};
  Eigen::Vector3d ang_vel = {0.0, 0.0, 1.0};

  const double delta_t = 0.005;
  
  const int N = 20;
  // integrate
  std::vector<IMUData> imu_data_vec;
  std::vector<IntegratedState> states;
  Eigen::Matrix3d C_mr = C_0;
  Eigen::Vector3d r_rm_in_m = r_0;
  Eigen::Vector3d v_rm_in_m = v_0;
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  states.push_back(IntegratedState(C_mr, r_rm_in_m, v_rm_in_m, 0.0));
  for (int i = 1; i < N; ++i) {
    imu_data_vec.push_back(IMUData(i * delta_t, ang_vel, lin_acc));
    r_rm_in_m += v_rm_in_m * delta_t + 0.5 * ((C_mr * (lin_acc - ba)) + gravity) * delta_t * delta_t;
    v_rm_in_m += ((C_mr * (lin_acc - ba)) + gravity) * delta_t;
    C_mr = C_mr * lgmath::so3::vec2rot(delta_t * (ang_vel - bg));
    states.push_back(IntegratedState(C_mr, r_rm_in_m, v_rm_in_m, i * delta_t));
  }
  r_rm_in_m += v_rm_in_m * delta_t + 0.5 * ((C_mr * (lin_acc - ba)) + gravity) * delta_t * delta_t;
  v_rm_in_m += ((C_mr * (lin_acc - ba)) + gravity) * delta_t;
  C_mr = C_mr * lgmath::so3::vec2rot(delta_t * (ang_vel - bg));
  states.push_back(IntegratedState(C_mr, r_rm_in_m, v_rm_in_m, N * delta_t));

  Eigen::Matrix4d T_i = Eigen::Matrix4d::Identity();
  const int state_index = 9;
  T_i.block<3, 3>(0, 0) = states[state_index].C_mr;
  T_i.block<3, 1>(0, 3) = states[state_index].r_rm_in_m;
  const Eigen::Matrix3d C_i = states[state_index].C_mr;
  Eigen::Vector3d v_i = states[state_index].v_rm_in_m;
  auto transform_i = lgmath::se3::Transformation(T_i);
  const auto T_i_var = se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_i));
  const auto v_i_var = vspace::PreIntVelocityStateVar<3>::MakeShared(v_i, T_i_var);
  Eigen::Matrix<double, 6, 1> bias = Eigen::Matrix<double, 6, 1>::Zero();
  const auto bias_var = vspace::VSpaceStateVar<6>::MakeShared(bias);

  auto options = P2PGlobalSuperCostTerm::Options();
  options.gravity = gravity;
  
  const auto p2p_cost_term = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var, v_i_var, bias_var, options, imu_data_vec);
  p2p_cost_term->set_min_time(0.0);
  p2p_cost_term->set_max_time(N * delta_t);
  
  const auto states_out = p2p_cost_term->integrate_(true);
  assert(states.size() == states_out.size());
  // check the integration is what we expect
  {
    for (size_t i = 0; i < states.size(); ++i) {
      std::cout << "time index: " << i << std::endl;
      Eigen::Vector3d error = lgmath::so3::rot2vec(states_out[i].C_mr * states[i].C_mr.transpose());
      std::cout << error.transpose() << std::endl;
      EXPECT_LT(error.norm(), 1e-6);
      error = states_out[i].r_rm_in_m - states[i].r_rm_in_m;
      std::cout << error.transpose() << std::endl;
      EXPECT_LT(error.norm(), 1e-6);
      error = states_out[i].v_rm_in_m - states[i].v_rm_in_m;
      std::cout << error.transpose() << std::endl;
      EXPECT_LT(error.norm(), 1e-6);
    }
  }

  // check the Jacobians by comparing against numerical Jacobians.
  const double eps = 1.0e-5;

  for (int jac_check_index = 0; jac_check_index < (int)states.size(); ++jac_check_index) {
    std::cout << "**************************************************" << std::endl;
    std::cout << "JAC CHECK INDEX: " << jac_check_index << std::endl;
    const Eigen::Matrix<double, 6, 15> jac = states_out[jac_check_index].jacobian;
    {
      std::cout << "r:" << std::endl;
      Eigen::Matrix<double, 6, 3> ajac = jac.block<6, 3>(0, 0);
      std::cout << ajac << std::endl;
      std::cout << "--------------------------------------" << std::endl;
      Eigen::Matrix<double, 6, 3> njac = Eigen::Matrix<double, 6, 3>::Zero();

      for (int j = 0; j < 3; ++j) {
        Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
        xi1(j, 0) = eps;
        xi2(j, 0) = -eps;
        Eigen::Matrix4d T_1 = T_i;
        T_1.block<3, 1>(0, 3) += T_i.block<3, 3>(0, 0) * xi1;
        Eigen::Matrix4d T_2 = T_i;
        T_2.block<3, 1>(0, 3) += T_i.block<3, 3>(0, 0) * xi2;
        const auto T_i_var_mod1 =
            se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_1));
        const auto T_i_var_mod2 =
            se3::SE3StateVarGlobalPerturb::MakeShared(lgmath::se3::Transformation(T_2));
        
        const auto p2p_cost_term_mod1 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var_mod1, v_i_var, bias_var, options, imu_data_vec);
        const auto p2p_cost_term_mod2 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var_mod2, v_i_var, bias_var, options, imu_data_vec);
        p2p_cost_term_mod1->set_min_time(0.0);
        p2p_cost_term_mod1->set_max_time(N * delta_t);
        p2p_cost_term_mod2->set_min_time(0.0);
        p2p_cost_term_mod2->set_max_time(N * delta_t);
        const auto states_mod1 = p2p_cost_term_mod1->integrate_(true);
        const auto states_mod2 = p2p_cost_term_mod2->integrate_(true);
        // trans
        njac.block<3, 1>(0, j) = (states_mod1[jac_check_index].r_rm_in_m - states_mod2[jac_check_index].r_rm_in_m) / (2 * eps);
        // rot
        njac.block<3, 1>(3, j) = (lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod1[jac_check_index].C_mr) - lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod2[jac_check_index].C_mr)) / (2 * eps);
      }
      std::cout << njac << std::endl;
      for (int i = 0; i < njac.rows(); ++i) {
        for (int j = 0; j < njac.cols(); ++j) {
          if (fabs(ajac(i, j)) > 1.0e-2)
            EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
          else
            EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
        }
      }
    }
    {
      std::cout << "C:" << std::endl;
      Eigen::Matrix<double, 6, 3> ajac = jac.block<6, 3>(0, 3);
      std::cout << ajac << std::endl;
      std::cout << "--------------------------------------" << std::endl;
      Eigen::Matrix<double, 6, 3> njac = Eigen::Matrix<double, 6, 3>::Zero();

      for (int j = 0; j < 3; ++j) {
        Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
        xi1(j, 0) = eps;
        xi2(j, 0) = -eps;
        Eigen::Matrix4d T_1 = Eigen::Matrix4d::Identity();
        T_1.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi1);
        Eigen::Matrix4d T_2 = Eigen::Matrix4d::Identity();
        T_2.block<3, 3>(0, 0) = lgmath::so3::vec2rot(xi2);
        const auto T_i_var_mod1 =
            se3::SE3StateVarGlobalPerturb::MakeShared(T_i * lgmath::se3::Transformation(T_1));
        const auto T_i_var_mod2 =
            se3::SE3StateVarGlobalPerturb::MakeShared(T_i * lgmath::se3::Transformation(T_2));
        
        const auto p2p_cost_term_mod1 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var_mod1, v_i_var, bias_var, options, imu_data_vec);
        const auto p2p_cost_term_mod2 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var_mod2, v_i_var, bias_var, options, imu_data_vec);
        p2p_cost_term_mod1->set_min_time(0.0);
        p2p_cost_term_mod1->set_max_time(N * delta_t);
        p2p_cost_term_mod2->set_min_time(0.0);
        p2p_cost_term_mod2->set_max_time(N * delta_t);
        const auto states_mod1 = p2p_cost_term_mod1->integrate_(true);
        const auto states_mod2 = p2p_cost_term_mod2->integrate_(true);
        // trans
        njac.block<3, 1>(0, j) = (states_mod1[jac_check_index].r_rm_in_m - states_mod2[jac_check_index].r_rm_in_m) / (2 * eps);
        // rot
        njac.block<3, 1>(3, j) = (lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod1[jac_check_index].C_mr) - lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod2[jac_check_index].C_mr)) / (2 * eps);
      }
      std::cout << njac << std::endl;
      for (int i = 0; i < njac.rows(); ++i) {
        for (int j = 0; j < njac.cols(); ++j) {
          if (fabs(ajac(i, j)) > 1.0e-2)
            EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
          else
            EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
        }
      }
    }
    {
      std::cout << "v:" << std::endl;
      Eigen::Matrix<double, 6, 3> ajac = jac.block<6, 3>(0, 6);
      std::cout << ajac << std::endl;
      std::cout << "--------------------------------------" << std::endl;
      Eigen::Matrix<double, 6, 3> njac = Eigen::Matrix<double, 6, 3>::Zero();

      for (int j = 0; j < 3; ++j) {
        Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
        xi1(j, 0) = eps;
        xi2(j, 0) = -eps;
        Eigen::Vector3d v_1 = v_i + C_i * xi1;
        Eigen::Vector3d v_2 = v_i + C_i * xi2;

        const auto v_i_var_mod1 = vspace::PreIntVelocityStateVar<3>::MakeShared(v_1, T_i_var);
        const auto v_i_var_mod2 = vspace::PreIntVelocityStateVar<3>::MakeShared(v_2, T_i_var);
        
        const auto p2p_cost_term_mod1 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var, v_i_var_mod1, bias_var, options, imu_data_vec);
        const auto p2p_cost_term_mod2 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var, v_i_var_mod2, bias_var, options, imu_data_vec);
        p2p_cost_term_mod1->set_min_time(0.0);
        p2p_cost_term_mod1->set_max_time(N * delta_t);
        p2p_cost_term_mod2->set_min_time(0.0);
        p2p_cost_term_mod2->set_max_time(N * delta_t);
        const auto states_mod1 = p2p_cost_term_mod1->integrate_(true);
        const auto states_mod2 = p2p_cost_term_mod2->integrate_(true);
        // trans
        njac.block<3, 1>(0, j) = (states_mod1[jac_check_index].r_rm_in_m - states_mod2[jac_check_index].r_rm_in_m) / (2 * eps);
        // rot
        njac.block<3, 1>(3, j) = (lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod1[jac_check_index].C_mr) - lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod2[jac_check_index].C_mr)) / (2 * eps);
      }
      std::cout << njac << std::endl;
      for (int i = 0; i < njac.rows(); ++i) {
        for (int j = 0; j < njac.cols(); ++j) {
          if (fabs(ajac(i, j)) > 1.0e-2)
            EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
          else
            EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
        }
      }
    }
    {
      std::cout << "ba:" << std::endl;
      Eigen::Matrix<double, 6, 3> ajac = jac.block<6, 3>(0, 9);
      std::cout << ajac << std::endl;
      std::cout << "--------------------------------------" << std::endl;
      Eigen::Matrix<double, 6, 3> njac = Eigen::Matrix<double, 6, 3>::Zero();

      for (int j = 0; j < 3; ++j) {
        Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
        xi1(j, 0) = eps;
        xi2(j, 0) = -eps;
        Eigen::Matrix<double, 6, 1> b_1 = bias;
        Eigen::Matrix<double, 6, 1> b_2 = bias;
        b_1.block<3, 1>(0, 0) += xi1;
        b_2.block<3, 1>(0, 0) += xi2;

        const auto bias_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(b_1);
        const auto bias_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(b_2);
        
        const auto p2p_cost_term_mod1 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var, v_i_var, bias_var_mod1, options, imu_data_vec);
        const auto p2p_cost_term_mod2 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var, v_i_var, bias_var_mod2, options, imu_data_vec);
        p2p_cost_term_mod1->set_min_time(0.0);
        p2p_cost_term_mod1->set_max_time(N * delta_t);
        p2p_cost_term_mod2->set_min_time(0.0);
        p2p_cost_term_mod2->set_max_time(N * delta_t);
        const auto states_mod1 = p2p_cost_term_mod1->integrate_(true);
        const auto states_mod2 = p2p_cost_term_mod2->integrate_(true);
        // trans
        njac.block<3, 1>(0, j) = (states_mod1[jac_check_index].r_rm_in_m - states_mod2[jac_check_index].r_rm_in_m) / (2 * eps);
        // rot
        njac.block<3, 1>(3, j) = (lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod1[jac_check_index].C_mr) - lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod2[jac_check_index].C_mr)) / (2 * eps);
      }
      std::cout << njac << std::endl;
      for (int i = 0; i < njac.rows(); ++i) {
        for (int j = 0; j < njac.cols(); ++j) {
          if (fabs(ajac(i, j)) > 1.0e-2)
            EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
          else
            EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
        }
      }
    }
    {
      std::cout << "bg:" << std::endl;
      Eigen::Matrix<double, 6, 3> ajac = jac.block<6, 3>(0, 12);
      std::cout << ajac << std::endl;
      std::cout << "--------------------------------------" << std::endl;
      Eigen::Matrix<double, 6, 3> njac = Eigen::Matrix<double, 6, 3>::Zero();

      for (int j = 0; j < 3; ++j) {
        Eigen::Matrix<double, 3, 1> xi1 = Eigen::Matrix<double, 3, 1>::Zero();
        Eigen::Matrix<double, 3, 1> xi2 = Eigen::Matrix<double, 3, 1>::Zero();
        xi1(j, 0) = eps;
        xi2(j, 0) = -eps;
        Eigen::Matrix<double, 6, 1> b_1 = bias;
        Eigen::Matrix<double, 6, 1> b_2 = bias;
        b_1.block<3, 1>(3, 0) += xi1;
        b_2.block<3, 1>(3, 0) += xi2;

        const auto bias_var_mod1 = vspace::VSpaceStateVar<6>::MakeShared(b_1);
        const auto bias_var_mod2 = vspace::VSpaceStateVar<6>::MakeShared(b_2);
        
        const auto p2p_cost_term_mod1 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var, v_i_var, bias_var_mod1, options, imu_data_vec);
        const auto p2p_cost_term_mod2 = P2PGlobalSuperCostTerm::MakeShared(Time(states[state_index].timestamp), T_i_var, v_i_var, bias_var_mod2, options, imu_data_vec);
        p2p_cost_term_mod1->set_min_time(0.0);
        p2p_cost_term_mod1->set_max_time(N * delta_t);
        p2p_cost_term_mod2->set_min_time(0.0);
        p2p_cost_term_mod2->set_max_time(N * delta_t);
        const auto states_mod1 = p2p_cost_term_mod1->integrate_(true);
        const auto states_mod2 = p2p_cost_term_mod2->integrate_(true);
        // trans
        njac.block<3, 1>(0, j) = (states_mod1[jac_check_index].r_rm_in_m - states_mod2[jac_check_index].r_rm_in_m) / (2 * eps);
        // rot
        njac.block<3, 1>(3, j) = (lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod1[jac_check_index].C_mr) - lgmath::so3::rot2vec(states[jac_check_index].C_mr.transpose() * states_mod2[jac_check_index].C_mr)) / (2 * eps);
      }
      std::cout << njac << std::endl;
      for (int i = 0; i < njac.rows(); ++i) {
        for (int j = 0; j < njac.cols(); ++j) {
          if (fabs(ajac(i, j)) > 1.0e-2)
            EXPECT_LT(fabs((njac(i, j) - ajac(i, j)) / ajac(i, j)), 0.01);
          else
            EXPECT_LT(fabs(njac(i, j) - ajac(i, j)), 1.0e-6);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
