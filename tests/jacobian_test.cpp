#include <gtest/gtest.h>

#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"

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
  const Eigen::Vector3d delta_q =
      beta * abar * abar.transpose() *
      lgmath::se3::point2fs(query).block<3, 6>(0, 0) *
      lgmath::se3::tranAd(T_vs.inverse()) * w_m_v_in_v;
  const Eigen::Vector3d reference =
      T_ms.block<3, 3>(0, 0) * (query + delta_q) + T_ms.block<3, 1>(0, 3);

  const auto T_vm_var = se3::SE3StateVar::MakeShared(T_vm);
  const auto w_mv_in_v_var = vspace::VSpaceStateVar<6>::MakeShared(w_m_v_in_v);

  const auto p2p_err_eval =
      p2pErrorDoppler(T_vm_var, w_mv_in_v_var, reference, query, beta);

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
          p2pErrorDoppler(T_vm_var_mod1, w_mv_in_v_var, reference, query, beta);
      const auto p2p_err_eval_mod2 =
          p2pErrorDoppler(T_vm_var_mod2, w_mv_in_v_var, reference, query, beta);

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
          p2pErrorDoppler(T_vm_var, w_mv_in_v_var_mod1, reference, query, beta);
      const auto p2p_err_eval_mod2 =
          p2pErrorDoppler(T_vm_var, w_mv_in_v_var_mod2, reference, query, beta);

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

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
