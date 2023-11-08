#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include <iostream>

namespace steam {

IMUSuperCostTerm::Ptr IMUSuperCostTerm::MakeShared(
    const Interface::ConstPtr &interface, const Time &time1, const Time &time2,
    const Evaluable<BiasType>::ConstPtr &bias1,
    const Evaluable<BiasType>::ConstPtr &bias2,
    const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
    const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
    const Options &options) {
  return std::make_shared<IMUSuperCostTerm>(interface, time1, time2, bias1,
                                            bias2, transform_i_to_m_1,
                                            transform_i_to_m_2, options);
}

/** \brief Compute the cost to the objective function */
double IMUSuperCostTerm::cost() const {
  double cost = 0;
  if (!frozen_) {
    using namespace steam::se3;
    using namespace steam::vspace;
    const auto T1_ = knot1_->pose()->forward();
    const auto w1_ = knot1_->velocity()->forward();
    const auto dw1_ = knot1_->acceleration()->forward();
    const auto T2_ = knot2_->pose()->forward();
    const auto w2_ = knot2_->velocity()->forward();
    const auto dw2_ = knot2_->acceleration()->forward();
    const auto b1_ = bias1_->forward();
    const auto b2_ = bias2_->forward();
    const auto T_mi_1_ = transform_i_to_m_1_->forward();
    const auto T_mi_2_ = transform_i_to_m_2_->forward();

    const auto T1 = T1_->value();
    const auto w1 = w1_->value();
    const auto dw1 = dw1_->value();
    const auto T2 = T2_->value();
    const auto w2 = w2_->value();
    const auto dw2 = dw2_->value();
    const auto b1 = b1_->value();
    const auto b2 = b2_->value();
    const auto T_mi_1 = T_mi_1_->value();
    const auto T_mi_2 = T_mi_2_->value();

    const auto xi_21 = (T2 / T1).vec();
    const lgmath::se3::Transformation T_21(xi_21);
    const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
    const auto J_21_inv_w2 = J_21_inv * w2;
    const auto J_21_inv_curl_dw2 =
        (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : cost)
    for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
      const double &ts = imu_data_vec_[i].timestamp;
      const IMUData &imu_data = imu_data_vec_[i];

      // pose, velocity, acceleration interpolation
      const auto &omega = interp_mats_.at(ts).first;
      const auto &lambda = interp_mats_.at(ts).second;
      const Eigen::Matrix<double, 6, 1> xi_i1 =
          lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
          omega.block<6, 6>(0, 0) * xi_21 +
          omega.block<6, 6>(0, 6) * J_21_inv_w2 +
          omega.block<6, 6>(0, 12) * J_21_inv_curl_dw2;
      const Eigen::Matrix<double, 6, 1> xi_j1 =
          lambda.block<6, 6>(6, 6) * w1 + lambda.block<6, 6>(6, 12) * dw1 +
          omega.block<6, 6>(6, 0) * xi_21 +
          omega.block<6, 6>(6, 6) * J_21_inv_w2 +
          omega.block<6, 6>(6, 12) * J_21_inv_curl_dw2;
      const Eigen::Matrix<double, 6, 1> xi_k1 =
          lambda.block<6, 6>(12, 6) * w1 + lambda.block<6, 6>(12, 12) * dw1 +
          omega.block<6, 6>(12, 0) * xi_21 +
          omega.block<6, 6>(12, 6) * J_21_inv_w2 +
          omega.block<6, 6>(12, 12) * J_21_inv_curl_dw2;

      // Interpolated pose
      const lgmath::se3::Transformation T_i1(xi_i1);
      const lgmath::se3::Transformation T_i0 = T_i1 * T1;
      // Interpolated velocity
      const Eigen::Matrix<double, 6, 1> w_i =
          lgmath::se3::vec2jac(xi_i1) * xi_j1;
      // Interpolated acceleration
      const Eigen::Matrix<double, 6, 1> dw_i =
          lgmath::se3::vec2jac(xi_i1) *
          (xi_k1 + 0.5 * lgmath::se3::curlyhat(xi_j1) * w_i);

      // Interpolated bias
      Eigen::Matrix<double, 6, 1> bias_i = Eigen::Matrix<double, 6, 1>::Zero();
      {
        const double tau = ts - knot1_->time().seconds();
        const double T = knot2_->time().seconds() - knot1_->time().seconds();
        const double ratio = tau / T;
        const double omega_ = ratio;
        const double lambda_ = 1 - ratio;
        bias_i = lambda_ * b1 + omega_ * b2;
      }

      // Interpolated T_mi
      lgmath::se3::Transformation transform_i_to_m;
      {
        const double alpha_ =
            (ts - knot1_->time().seconds()) /
            (knot2_->time().seconds() - knot1_->time().seconds());
        const Eigen::Matrix<double, 6, 1> xi_i1_ =
            alpha_ * (T_mi_2 / T_mi_1).vec();
        transform_i_to_m = lgmath::se3::Transformation(xi_i1_) * T_mi_1;
      }

      const Eigen::Matrix3d C_vm = T_i0.matrix().block<3, 3>(0, 0);
      const Eigen::Matrix3d C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);

      const Eigen::Matrix<double, 3, 1> raw_error_acc =
          imu_data.lin_acc + dw_i.block<3, 1>(0, 0) +
          C_vm * C_mi * options_.gravity - bias_i.block<3, 1>(0, 0);

      cost += acc_loss_func_->cost(
          acc_noise_model_->getWhitenedErrorNorm(raw_error_acc));

      const Eigen::Matrix<double, 3, 1> raw_error_gyro =
          imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);

      cost += gyro_loss_func_->cost(
          gyro_noise_model_->getWhitenedErrorNorm(raw_error_gyro));
    }
  }
  return cost;
}

/** \brief Get keys of variables related to this cost term */
void IMUSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot1_->acceleration()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
  knot2_->acceleration()->getRelatedVarKeys(keys);
  bias1_->getRelatedVarKeys(keys);
  bias2_->getRelatedVarKeys(keys);
  transform_i_to_m_1_->getRelatedVarKeys(keys);
  transform_i_to_m_2_->getRelatedVarKeys(keys);
}

void IMUSuperCostTerm::init() { initialize_interp_matrices_(); }

void IMUSuperCostTerm::initialize_interp_matrices_() {
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
  for (const IMUData &imu_data : imu_data_vec_) {
    const double &time = imu_data.timestamp;
    // const auto &time = it->first;
    if (interp_mats_.find(time) == interp_mats_.end()) {
      // Get Lambda, Omega for this time
      const double tau = time - time1_.seconds();
      const double kappa = knot2_->time().seconds() - time;
      const Matrix18d Q_tau = interface_->getQPublic(tau, ones);
      const Matrix18d Tran_kappa = interface_->getTranPublic(kappa);
      const Matrix18d Tran_tau = interface_->getTranPublic(tau);
      const Matrix18d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
      const Matrix18d lambda = (Tran_tau - omega * Tran_T_);
      interp_mats_.emplace(time, std::make_pair(omega, lambda));
    }
  }
}

void IMUSuperCostTerm::buildGaussNewtonTerms_(
    Eigen::Matrix<double, 60, 60> &A, Eigen::Matrix<double, 60, 1> &b) const {
  using namespace steam::se3;
  using namespace steam::vspace;
  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto dw1_ = knot1_->acceleration()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();
  const auto dw2_ = knot2_->acceleration()->forward();
  const auto b1_ = bias1_->forward();
  const auto b2_ = bias2_->forward();
  const auto T_mi_1_ = transform_i_to_m_1_->forward();
  const auto T_mi_2_ = transform_i_to_m_2_->forward();

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto dw1 = dw1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();
  const auto dw2 = dw2_->value();
  const auto b1 = b1_->value();
  const auto b2 = b2_->value();
  const auto T_mi_1 = T_mi_1_->value();
  const auto T_mi_2 = T_mi_2_->value();

  A = A_;
  b = b_;

  if (!frozen_) {
    const auto xi_21 = (T2 / T1).vec();
    const lgmath::se3::Transformation T_21(xi_21);
    const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
    const auto J_21_inv_w2 = J_21_inv * w2;
    const auto J_21_inv_curl_dw2 =
        (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

    // If some variables are not active? (simply don't use those parts
    // of the A, b to update hessian, grad at the end)
    A = Eigen::Matrix<double, 60, 60>::Zero();
    b = Eigen::Matrix<double, 60, 1>::Zero();
#pragma omp declare reduction(+ : Eigen::Matrix<double, 60, 60> : omp_out = \
                                  omp_out + omp_in)                         \
    initializer(omp_priv = Eigen::Matrix<double, 60, 60>::Zero())
#pragma omp declare reduction(+ : Eigen::Matrix<double, 60, 1> : omp_out = \
                                  omp_out + omp_in)                        \
    initializer(omp_priv = Eigen::Matrix<double, 60, 1>::Zero())
#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : A) \
    reduction(+ : b)
    for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
      const double &ts = imu_data_vec_[i].timestamp;
      const IMUData &imu_data = imu_data_vec_[i];

      // pose, velocity, acceleration interpolation
      const auto &omega = interp_mats_.at(ts).first;
      const auto &lambda = interp_mats_.at(ts).second;
      const Eigen::Matrix<double, 6, 1> xi_i1 =
          lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
          omega.block<6, 6>(0, 0) * xi_21 +
          omega.block<6, 6>(0, 6) * J_21_inv_w2 +
          omega.block<6, 6>(0, 12) * J_21_inv_curl_dw2;
      const Eigen::Matrix<double, 6, 1> xi_j1 =
          lambda.block<6, 6>(6, 6) * w1 + lambda.block<6, 6>(6, 12) * dw1 +
          omega.block<6, 6>(6, 0) * xi_21 +
          omega.block<6, 6>(6, 6) * J_21_inv_w2 +
          omega.block<6, 6>(6, 12) * J_21_inv_curl_dw2;
      const Eigen::Matrix<double, 6, 1> xi_k1 =
          lambda.block<6, 6>(12, 6) * w1 + lambda.block<6, 6>(12, 12) * dw1 +
          omega.block<6, 6>(12, 0) * xi_21 +
          omega.block<6, 6>(12, 6) * J_21_inv_w2 +
          omega.block<6, 6>(12, 12) * J_21_inv_curl_dw2;

      // Interpolated pose
      const lgmath::se3::Transformation T_i1(xi_i1);
      const lgmath::se3::Transformation T_i0 = T_i1 * T1;
      // Interpolated velocity
      const Eigen::Matrix<double, 6, 1> w_i =
          lgmath::se3::vec2jac(xi_i1) * xi_j1;
      // Interpolated acceleration
      const Eigen::Matrix<double, 6, 1> dw_i =
          lgmath::se3::vec2jac(xi_i1) *
          (xi_k1 + 0.5 * lgmath::se3::curlyhat(xi_j1) * w_i);

      // Interpolated bias
      Eigen::Matrix<double, 6, 1> bias_i = Eigen::Matrix<double, 6, 1>::Zero();
      Eigen::Matrix<double, 6, 12> interp_jac_bias =
          Eigen::Matrix<double, 6, 12>::Zero();
      {
        const double tau = ts - knot1_->time().seconds();
        const double T = knot2_->time().seconds() - knot1_->time().seconds();
        const double ratio = tau / T;
        const double omega_ = ratio;
        const double lambda_ = 1 - ratio;
        bias_i = lambda_ * b1 + omega_ * b2;
        interp_jac_bias.block<6, 6>(0, 0) =
            Eigen::Matrix<double, 6, 6>::Identity() * lambda_;
        interp_jac_bias.block<6, 6>(0, 6) =
            Eigen::Matrix<double, 6, 6>::Identity() * omega_;
      }

      // Interpolated T_mi
      lgmath::se3::Transformation transform_i_to_m;
      Eigen::Matrix<double, 6, 12> interp_jac_T_m_i =
          Eigen::Matrix<double, 6, 12>::Zero();
      {
        const double alpha_ =
            (ts - knot1_->time().seconds()) /
            (knot2_->time().seconds() - knot1_->time().seconds());
        const Eigen::Matrix<double, 6, 1> xi_i1_ =
            alpha_ * (T_mi_2 / T_mi_1).vec();
        transform_i_to_m = lgmath::se3::Transformation(xi_i1_) * T_mi_1;
        if (transform_i_to_m_1_->active() || transform_i_to_m_2_->active()) {
          std::vector<double> faulhaber_coeffs_;
          faulhaber_coeffs_.push_back(alpha_);
          faulhaber_coeffs_.push_back(alpha_ * (alpha_ - 1) / 2);
          faulhaber_coeffs_.push_back(alpha_ * (alpha_ - 1) * (2 * alpha_ - 1) /
                                      12);
          faulhaber_coeffs_.push_back(alpha_ * alpha_ * (alpha_ - 1) *
                                      (alpha_ - 1) / 24);
          const Eigen::Matrix<double, 6, 6> xi_21_curlyhat =
              lgmath::se3::curlyhat((T_mi_2 / T_mi_1).vec());
          Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
          Eigen::Matrix<double, 6, 6> xictmp =
              Eigen::Matrix<double, 6, 6>::Identity();
          for (unsigned int i = 0; i < faulhaber_coeffs_.size(); i++) {
            if (i > 0) xictmp = xi_21_curlyhat * xictmp;
            A += faulhaber_coeffs_[i] * xictmp;
          }
          interp_jac_T_m_i.block<6, 6>(0, 0) =
              Eigen::Matrix<double, 6, 6>::Identity() - A;
          interp_jac_T_m_i.block<6, 6>(0, 6) = A;
        }
      }

      // pose interpolation Jacobian
      Eigen::Matrix<double, 6, 36> interp_jac_pose =
          Eigen::Matrix<double, 6, 36>::Zero();
      // velocity interpolation Jacobian
      Eigen::Matrix<double, 6, 36> interp_jac_vel =
          Eigen::Matrix<double, 6, 36>::Zero();
      // acceleration interpolation Jacobian
      Eigen::Matrix<double, 6, 36> interp_jac_acc =
          Eigen::Matrix<double, 6, 36>::Zero();

      const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);

      const Eigen::Matrix<double, 6, 6> xi_j1_ch =
          -0.5 * lgmath::se3::curlyhat(xi_j1);

      const auto J_prep_2 = J_i1 * (-0.5 * lgmath::se3::curlyhat(w_i) +
                                    0.5 * lgmath::se3::curlyhat(xi_j1) * J_i1);
      const auto J_prep_3 =
          -0.25 * J_i1 * lgmath::se3::curlyhat(xi_j1) *
              lgmath::se3::curlyhat(xi_j1) -
          0.5 * lgmath::se3::curlyhat(xi_k1 +
                                      0.5 * lgmath::se3::curlyhat(xi_j1) * w_i);

      // pose interpolation Jacobian
      Eigen::Matrix<double, 6, 6> w =
          J_i1 *
          (omega.block<6, 6>(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() +
           omega.block<6, 6>(0, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
           omega.block<6, 6>(0, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
               lgmath::se3::curlyhat(w2) +
           omega.block<6, 6>(0, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
          J_21_inv;

      interp_jac_pose.block<6, 6>(0, 0) =
          -w * T_21.adjoint() + T_i1.adjoint();  // T1
      interp_jac_pose.block<6, 6>(0, 6) =
          lambda.block<6, 6>(0, 6) * J_i1;  // w1
      interp_jac_pose.block<6, 6>(0, 12) =
          lambda.block<6, 6>(0, 12) * J_i1;    // dw1
      interp_jac_pose.block<6, 6>(0, 18) = w;  // T2
      interp_jac_pose.block<6, 6>(0, 24) =
          omega.block<6, 6>(0, 6) * J_i1 * J_21_inv +
          omega.block<6, 6>(0, 12) * -0.5 * J_i1 *
              (lgmath::se3::curlyhat(J_21_inv * w2) -
               lgmath::se3::curlyhat(w2) * J_21_inv);  // w2
      interp_jac_pose.block<6, 6>(0, 30) =
          omega.block<6, 6>(0, 12) * J_i1 * J_21_inv;  // dw2

      // velocity interpolation Jacobian
      w = J_i1 *
              (omega.block<6, 6>(6, 0) *
                   Eigen::Matrix<double, 6, 6>::Identity() +
               omega.block<6, 6>(6, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(6, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
                   lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(6, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
              J_21_inv +
          xi_j1_ch *
              (omega.block<6, 6>(0, 0) *
                   Eigen::Matrix<double, 6, 6>::Identity() +
               omega.block<6, 6>(0, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(0, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
                   lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(0, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
              J_21_inv;

      interp_jac_vel.block<6, 6>(0, 0) = -w * T_21.adjoint();  // T1
      interp_jac_vel.block<6, 6>(0, 6) =
          J_i1 * lambda.block<6, 6>(6, 6) +
          xi_j1_ch * lambda.block<6, 6>(0, 6);  // w1
      interp_jac_vel.block<6, 6>(0, 12) =
          J_i1 * lambda.block<6, 6>(6, 12) +
          xi_j1_ch * lambda.block<6, 6>(0, 12);  // dw1
      interp_jac_vel.block<6, 6>(0, 18) = w;     // T2
      interp_jac_vel.block<6, 6>(0, 24) =
          J_i1 * (omega.block<6, 6>(6, 6) * J_21_inv +
                  omega.block<6, 6>(6, 12) * -0.5 *
                      (lgmath::se3::curlyhat(J_21_inv * w2) -
                       lgmath::se3::curlyhat(w2) * J_21_inv)) +
          xi_j1_ch * (omega.block<6, 6>(0, 6) * J_21_inv +
                      omega.block<6, 6>(0, 12) * -0.5 *
                          (lgmath::se3::curlyhat(J_21_inv * w2) -
                           lgmath::se3::curlyhat(w2) * J_21_inv));  // w2
      interp_jac_vel.block<6, 6>(0, 30) =
          J_i1 * (omega.block<6, 6>(6, 12) * J_21_inv) +
          xi_j1_ch * (omega.block<6, 6>(0, 12) * J_21_inv);  // dw2

      // acceleration interpolation Jacobian
      w = J_i1 *
              (omega.block<6, 6>(12, 0) *
                   Eigen::Matrix<double, 6, 6>::Identity() +
               omega.block<6, 6>(12, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(12, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
                   lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(12, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
              J_21_inv +
          J_prep_2 *
              (omega.block<6, 6>(6, 0) *
                   Eigen::Matrix<double, 6, 6>::Identity() +
               omega.block<6, 6>(6, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(6, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
                   lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(6, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
              J_21_inv +
          J_prep_3 *
              (omega.block<6, 6>(0, 0) *
                   Eigen::Matrix<double, 6, 6>::Identity() +
               omega.block<6, 6>(0, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(0, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
                   lgmath::se3::curlyhat(w2) +
               omega.block<6, 6>(0, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
              J_21_inv;

      interp_jac_acc.block<6, 6>(0, 0) = -w * T_21.adjoint();  // T1
      interp_jac_acc.block<6, 6>(0, 6) =
          J_i1 * lambda.block<6, 6>(12, 6) +
          J_prep_2 * lambda.block<6, 6>(6, 6) +
          J_prep_3 * lambda.block<6, 6>(0, 6);  // w1
      interp_jac_acc.block<6, 6>(0, 12) =
          J_i1 * lambda.block<6, 6>(12, 12) +
          J_prep_2 * lambda.block<6, 6>(6, 12) +
          J_prep_3 * lambda.block<6, 6>(0, 12);  // dw1
      interp_jac_acc.block<6, 6>(0, 18) = w;     // T2
      interp_jac_acc.block<6, 6>(0, 24) =
          J_i1 * (omega.block<6, 6>(12, 6) * J_21_inv +
                  omega.block<6, 6>(12, 12) * -0.5 *
                      (lgmath::se3::curlyhat(J_21_inv * w2) -
                       lgmath::se3::curlyhat(w2) * J_21_inv)) +
          J_prep_2 * (omega.block<6, 6>(6, 6) * J_21_inv +
                      omega.block<6, 6>(6, 12) * -0.5 *
                          (lgmath::se3::curlyhat(J_21_inv * w2) -
                           lgmath::se3::curlyhat(w2) * J_21_inv)) +
          J_prep_3 * (omega.block<6, 6>(0, 6) * J_21_inv +
                      omega.block<6, 6>(0, 12) * -0.5 *
                          (lgmath::se3::curlyhat(J_21_inv * w2) -
                           lgmath::se3::curlyhat(w2) * J_21_inv));  // w2
      interp_jac_acc.block<6, 6>(0, 30) =
          J_i1 * (omega.block<6, 6>(12, 12) * J_21_inv) +
          J_prep_2 * (omega.block<6, 6>(6, 12) * J_21_inv) +
          J_prep_3 * (omega.block<6, 6>(0, 12) * J_21_inv);  // dw2

      const Eigen::Matrix3d C_vm = T_i0.matrix().block<3, 3>(0, 0);
      const Eigen::Matrix3d C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);

      // get measurement Jacobians
      const Eigen::Matrix<double, 3, 1> raw_error_acc =
          imu_data.lin_acc + dw_i.block<3, 1>(0, 0) +
          C_vm * C_mi * options_.gravity - bias_i.block<3, 1>(0, 0);
      const Eigen::Matrix<double, 3, 1> white_error_acc =
          acc_noise_model_->whitenError(raw_error_acc);
      const double sqrt_w_acc =
          sqrt(acc_loss_func_->weight(white_error_acc.norm()));
      const Eigen::Matrix<double, 3, 1> error_acc =
          sqrt_w_acc * white_error_acc;

      // Todo: weight and whiten errors and Jacobians

      Eigen::Matrix<double, 3, 24> Gmeas = Eigen::Matrix<double, 3, 24>::Zero();
      // Acceleration measurement Jacobians
      Gmeas.block<3, 3>(0, 3) =
          -1 * lgmath::so3::hat(C_vm * C_mi * options_.gravity);
      Gmeas.block<3, 6>(0, 6) = jac_accel_;
      Gmeas.block<3, 6>(0, 12) = jac_bias_accel_;
      Gmeas.block<3, 3>(0, 21) =
          -1 * C_vm * lgmath::so3::hat(C_mi * options_.gravity);

      Gmeas = sqrt_w_acc * acc_noise_model_->getSqrtInformation() * Gmeas;

      Eigen::Matrix<double, 6, 60> G = Eigen::Matrix<double, 6, 60>::Zero();
      G.block<3, 36>(0, 0) = Gmeas.block<3, 6>(0, 0) * interp_jac_pose +
                             Gmeas.block<3, 6>(0, 6) * interp_jac_acc;
      G.block<3, 12>(0, 36) = Gmeas.block<3, 6>(0, 12) * interp_jac_bias;
      G.block<3, 12>(0, 48) = Gmeas.block<3, 6>(0, 18) * interp_jac_T_m_i;

      const Eigen::Matrix<double, 3, 1> raw_error_gyro =
          imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
      const Eigen::Matrix<double, 3, 1> white_error_gyro =
          gyro_noise_model_->whitenError(raw_error_gyro);
      const double sqrt_w_gyro =
          sqrt(gyro_loss_func_->weight(white_error_gyro.norm()));
      const Eigen::Matrix<double, 3, 1> error_gyro =
          sqrt_w_gyro * white_error_gyro;

      G.block<3, 36>(3, 0) = sqrt_w_gyro *
                             gyro_noise_model_->getSqrtInformation() *
                             jac_vel_ * interp_jac_vel;
      G.block<3, 12>(3, 36) = sqrt_w_gyro *
                              gyro_noise_model_->getSqrtInformation() *
                              jac_bias_gyro_ * interp_jac_bias;

      Eigen::Matrix<double, 6, 1> error = Eigen::Matrix<double, 6, 1>::Zero();
      error.block<3, 1>(0, 0) = error_acc;
      error.block<3, 1>(3, 0) = error_gyro;

      A += G.transpose() * G;
      b += (-1) * G.transpose() * error;
    }
  }
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void IMUSuperCostTerm::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  using namespace steam::se3;
  using namespace steam::vspace;
  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto dw1_ = knot1_->acceleration()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();
  const auto dw2_ = knot2_->acceleration()->forward();
  const auto b1_ = bias1_->forward();
  const auto b2_ = bias2_->forward();
  const auto T_mi_1_ = transform_i_to_m_1_->forward();
  const auto T_mi_2_ = transform_i_to_m_2_->forward();
  Eigen::Matrix<double, 60, 60> A = Eigen::Matrix<double, 60, 60>::Zero();
  Eigen::Matrix<double, 60, 1> b = Eigen::Matrix<double, 60, 1>::Zero();

  buildGaussNewtonTerms_(A, b);

  std::vector<bool> active;
  active.push_back(knot1_->pose()->active());
  active.push_back(knot1_->velocity()->active());
  active.push_back(knot1_->acceleration()->active());
  active.push_back(knot2_->pose()->active());
  active.push_back(knot2_->velocity()->active());
  active.push_back(knot2_->acceleration()->active());
  active.push_back(bias1_->active());
  active.push_back(bias2_->active());
  active.push_back(transform_i_to_m_1_->active());
  active.push_back(transform_i_to_m_2_->active());

  std::vector<StateKey> keys;

  if (active[0]) {
    const auto T1node = std::static_pointer_cast<Node<PoseType>>(T1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot1_->pose()->backward(lhs, T1node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[1]) {
    const auto w1node = std::static_pointer_cast<Node<VelType>>(w1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot1_->velocity()->backward(lhs, w1node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[2]) {
    const auto dw1node = std::static_pointer_cast<Node<AccType>>(dw1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot1_->acceleration()->backward(lhs, dw1node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[3]) {
    const auto T2node = std::static_pointer_cast<Node<PoseType>>(T2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot2_->pose()->backward(lhs, T2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[4]) {
    const auto w2node = std::static_pointer_cast<Node<VelType>>(w2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot2_->velocity()->backward(lhs, w2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[5]) {
    const auto dw2node = std::static_pointer_cast<Node<AccType>>(dw2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot2_->acceleration()->backward(lhs, dw2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[6]) {
    const auto b1node = std::static_pointer_cast<Node<BiasType>>(b1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    bias1_->backward(lhs, b1node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }

  } else {
    keys.push_back(-1);
  }
  if (active[7]) {
    const auto b2node = std::static_pointer_cast<Node<BiasType>>(b2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    bias2_->backward(lhs, b2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[8]) {
    const auto T_mi_1_node = std::static_pointer_cast<Node<PoseType>>(T_mi_1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    transform_i_to_m_1_->backward(lhs, T_mi_1_node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[9]) {
    const auto T_mi_2_node = std::static_pointer_cast<Node<PoseType>>(T_mi_2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    transform_i_to_m_2_->backward(lhs, T_mi_2_node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  // std::cout << "keys:";
  // for (unsigned int i = 0; i < keys.size(); ++i) {
  //   if (!active[i]) continue;
  //   std::cout << state_vec.getStateBlockIndex(keys[i]) << " ";
  // }
  // std::cout << std::endl;

  for (int i = 0; i < 10; ++i) {
    if (!active[i]) continue;
    // Get the key and state range affected
    const auto &key1 = keys[i];
    unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

    // Calculate terms needed to update the right-hand-side
    Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

    // Update the right-hand side (thread critical)

#pragma omp critical(b_update)
    { gradient_vector->mapAt(blkIdx1) += newGradTerm; }

    for (int j = i; j < 10; ++j) {
      if (!active[j]) continue;
      // Get the key and state range affected
      const auto &key2 = keys[j];
      unsigned int blkIdx2 = state_vec.getStateBlockIndex(key2);

      // Calculate terms needed to update the Gauss-Newton left-hand side
      unsigned int row, col;
      const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
        if (blkIdx1 <= blkIdx2) {
          row = blkIdx1;
          col = blkIdx2;
          return A.block<6, 6>(i * 6, j * 6);
        } else {
          row = blkIdx2;
          col = blkIdx1;
          return A.block<6, 6>(j * 6, i * 6);
        }
      }();

      // Update the left-hand side (thread critical)
      BlockSparseMatrix::BlockRowEntry &entry =
          approximate_hessian->rowEntryAt(row, col, true);
      omp_set_lock(&entry.lock);
      entry.data += newHessianTerm;
      omp_unset_lock(&entry.lock);
    }
  }
}  // namespace steam

}  // namespace steam
