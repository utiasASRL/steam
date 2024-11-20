#include "steam/problem/cost_term/preintegrated_accel_cost_term.hpp"
#include <iostream>

namespace steam {

PreintAccCostTerm::Ptr PreintAccCostTerm::MakeShared(
    const Interface::ConstPtr &interface, const Time time1, const Time time2,
    const Evaluable<BiasType>::ConstPtr &bias1,
    const Evaluable<BiasType>::ConstPtr &bias2,
    const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
    const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
    const Options &options) {
  return std::make_shared<PreintAccCostTerm>(interface, time1, time2, bias1,
                                             bias2, transform_i_to_m_1,
                                             transform_i_to_m_2, options);
}

/** \brief Compute the cost to the objective function */
double PreintAccCostTerm::cost() const {
  using namespace steam::se3;
  using namespace steam::vspace;
  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();
  const auto b1_ = bias1_->forward();
  const auto b2_ = bias2_->forward();
  const auto T_mi_1_ = transform_i_to_m_1_->forward();
  const auto T_mi_2_ = transform_i_to_m_2_->forward();

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();
  const auto b1 = b1_->value();
  const auto b2 = b2_->value();
  const auto T_mi_1 = T_mi_1_->value();
  const auto T_mi_2 = T_mi_2_->value();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto J_21_inv_w2 = J_21_inv * w2;

  Eigen::Vector3d preint_delta_v = Eigen::Vector3d::Zero();
  Eigen::Vector3d v1 = w1.block<3, 1>(0, 0);
  Eigen::Matrix3d R = Eigen::Matrix3d::Zero();

  // #pragma omp declare reduction(+ : Eigen::Vector3d : omp_out = \
  //                                 omp_out + omp_in)           \
  //   initializer(omp_priv = Eigen::Vector3d::Zero())
  // #pragma omp declare reduction(+ : Eigen::Matrix3d : omp_out = \
  //                                 omp_out + omp_in)           \
  //   initializer(omp_priv = Eigen::Matrix3d::Zero())
  // #pragma omp parallel for num_threads(options_.num_threads) \
  //   reduction(+ : preint_delta_v) reduction(+ : R)
  for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
    const double &ts = imu_data_vec_[i].timestamp;
    const IMUData &imu_data = imu_data_vec_[i];

    // pose interpolation
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda(0, 1) * w1 + omega(0, 0) * xi_21 +
        omega(0, 1) * J_21_inv_w2;
    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;

    const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);

    if (i == 0) {
      const Eigen::Matrix<double, 6, 1> xi_j1 =
          lambda(1, 1) * w1 + omega(1, 0) * xi_21 +
          omega(1, 1) * J_21_inv_w2;
      const Eigen::Matrix<double, 6, 1> w_i = J_i1 * xi_j1;
      v1 = w_i.block<3, 1>(0, 0);
    }

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
    lgmath::se3::Transformation transform_i_to_m = T_mi_1;
    if (transform_i_to_m_1_->active() && transform_i_to_m_2_->active()) {
      const double alpha_ =
          (ts - knot1_->time().seconds()) /
          (knot2_->time().seconds() - knot1_->time().seconds());
      const Eigen::Matrix<double, 6, 1> xi_i1_ =
          alpha_ * (T_mi_2 / T_mi_1).vec();
      transform_i_to_m = lgmath::se3::Transformation(xi_i1_) * T_mi_1;
    } else if (transform_i_to_m_1_->active() && !transform_i_to_m_2_->active()) {
      transform_i_to_m = T_mi_1;
    } else if (!transform_i_to_m_1_->active() && transform_i_to_m_2_->active()) {
      std::runtime_error("either (T_mi_1 and T_mi_2) need to be active or just T_mi_1");
    }

    const Eigen::Matrix3d &C_vm = T_i0.matrix().block<3, 3>(0, 0);
    const Eigen::Matrix3d &C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);

    double delta_ts = 0.;
    if (i < (int)imu_data_vec_.size() - 1) {
      delta_ts = imu_data_vec_[i + 1].timestamp - imu_data_vec_[i].timestamp;
    } else {
      delta_ts = knot2_->time().seconds() - imu_data_vec_[i].timestamp;
    }
    if (options_.se2) {
      preint_delta_v.block<2, 1>(0, 0) +=
          (imu_data.lin_acc.block<2, 1>(0, 0) - bias_i.block<2, 1>(0, 0)) *
          delta_ts;
    } else {
      preint_delta_v += (imu_data.lin_acc + C_vm * C_mi * options_.gravity -
                         bias_i.block<3, 1>(0, 0)) *
                        delta_ts;
    }
    R += R_acc_ * delta_ts * delta_ts;
  }

  Eigen::Vector3d raw_error = Eigen::Vector3d::Zero();
  if (options_.se2) {
    raw_error.block<2, 1>(0, 0) = w2.block<2, 1>(0, 0) - v1.block<2, 1>(0, 0) +
                                  preint_delta_v.block<2, 1>(0, 0);
  } else {
    raw_error = w2.block<3, 1>(0, 0) - v1 + preint_delta_v;
  }
  StaticNoiseModel<3>::Ptr noise_model = StaticNoiseModel<3>::MakeShared(R);
  const auto cost = loss_func_->cost(noise_model->getWhitenedErrorNorm(raw_error));
  return cost;
}

/** \brief Get keys of variables related to this cost term */
void PreintAccCostTerm::getRelatedVarKeys(KeySet &keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
  bias1_->getRelatedVarKeys(keys);
  bias2_->getRelatedVarKeys(keys);
  transform_i_to_m_1_->getRelatedVarKeys(keys);
  transform_i_to_m_2_->getRelatedVarKeys(keys);
}

void PreintAccCostTerm::init() { initialize_interp_matrices_(); }

void PreintAccCostTerm::initialize_interp_matrices_() {
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
// #pragma omp parallel for num_threads(options_.num_threads)
  for (const IMUData &imu_data : imu_data_vec_) {
    const double &time = imu_data.timestamp;
    if (interp_mats_.find(time) == interp_mats_.end()) {
      const double tau = (Time(time) - time1_).seconds();
      // const double T = (time2_ - time1_).seconds();
      // const double ratio = tau / T;
      // const double ratio2 = ratio * ratio;
      // const double ratio3 = ratio2 * ratio;
      // Calculate 'omega' interpolation values
      Eigen::Matrix4d omega = Eigen::Matrix4d::Zero();
      // omega(0, 0) = 3.0 * ratio2 - 2.0 * ratio3;
      // omega(0, 1) = tau * (ratio2 - ratio);
      // omega(1, 0) = 6.0 * (ratio - ratio2) / T;
      // omega(1, 1) = 3.0 * ratio2 - 2.0 * ratio;
      // Calculate 'lambda' interpolation values
      Eigen::Matrix4d lambda = Eigen::Matrix4d::Zero();
      // lambda(0, 0) = 1.0 - omega(0, 0);
      // lambda(0, 1) = tau - T * omega(0, 0) - omega(0, 1);
      // lambda(1, 0) = -omega(1, 0);
      // lambda(1, 1) = 1.0 - T * omega(1, 0) - omega(1, 1);
      const double kappa = knot2_->time().seconds() - time;
      const Matrix12d Q_tau = steam::traj::const_vel::getQ(tau, ones);
      const Matrix12d Tran_kappa = steam::traj::const_vel::getTran(kappa);
      const Matrix12d Tran_tau = steam::traj::const_vel::getTran(tau);
      const Matrix12d omega12 = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
      const Matrix12d lambda12 = (Tran_tau - omega12 * Tran_T_);
      omega(0, 0) = omega12(0, 0);
      omega(1, 0) = omega12(6, 0);
      omega(0, 1) = omega12(0, 6);
      omega(1, 1) = omega12(6, 6);
      lambda(0, 0) = lambda12(0, 0);
      lambda(1, 0) = lambda12(6, 0);
      lambda(0, 1) = lambda12(0, 6);
      lambda(1, 1) = lambda12(6, 6);
      
      interp_mats_.emplace(time, std::make_pair(omega, lambda));
    }
  }
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void PreintAccCostTerm::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  using namespace steam::se3;
  using namespace steam::vspace;

  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();
  const auto b1_ = bias1_->forward();
  const auto b2_ = bias2_->forward();
  const auto T_mi_1_ = transform_i_to_m_1_->forward();
  const auto T_mi_2_ = transform_i_to_m_2_->forward();

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();
  const auto b1 = b1_->value();
  const auto b2 = b2_->value();
  const auto T_mi_1 = T_mi_1_->value();
  const auto T_mi_2 = T_mi_2_->value();

  Eigen::Matrix<double, 48, 48> A = Eigen::Matrix<double, 48, 48>::Zero();
  Eigen::Matrix<double, 48, 1> b = Eigen::Matrix<double, 48, 1>::Zero();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const auto Ad_T_21 = lgmath::se3::tranAd(T_21.matrix());
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto w2_j_21_inv = 0.5 * lgmath::se3::curlyhat(w2) * J_21_inv;
  const auto J_21_inv_w2 = J_21_inv * w2;

  Eigen::Vector3d preint_delta_v = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 3, 48> G = Eigen::Matrix<double, 3, 48>::Zero();
  Eigen::Vector3d v1 = w1.block<3, 1>(0, 0);
  Eigen::Matrix<double, 3, 24> interp_jac_v1 =
      Eigen::Matrix<double, 3, 24>::Zero();
  Eigen::Matrix3d R = Eigen::Matrix3d::Zero();

  // #pragma omp declare reduction(+ : Eigen::Matrix<double, 3, 48> : omp_out = \
  //                                 omp_out + omp_in)                        \
  //   initializer(omp_priv = Eigen::Matrix<double, 3, 48>::Zero())
  // #pragma omp declare reduction(+ : Eigen::Vector3d : omp_out = \
  //                                 omp_out + omp_in)           \
  //   initializer(omp_priv = Eigen::Vector3d::Zero())
  // #pragma omp declare reduction(+ : Eigen::Matrix3d : omp_out = \
  //                                 omp_out + omp_in)           \
  //   initializer(omp_priv = Eigen::Matrix3d::Zero())
  // #pragma omp parallel for num_threads(options_.num_threads) reduction(+ : G) \
  //   reduction(+ : preint_delta_v) reduction(+ : R)
  for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
    const double &ts = imu_data_vec_[i].timestamp;
    const IMUData &imu_data = imu_data_vec_[i];

    // pose interpolation
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda(0, 1) * w1 + omega(0, 0) * xi_21 +
        omega(0, 1) * J_21_inv_w2;
    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;

    const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);

    if (i == 0) {
      const Eigen::Matrix<double, 6, 1> xi_j1 =
          lambda(1, 1) * w1 + omega(1, 0) * xi_21 +
          omega(1, 1) * J_21_inv_w2;
      const Eigen::Matrix<double, 6, 1> w_i = J_i1 * xi_j1;
      v1 = w_i.block<3, 1>(0, 0);
      const Eigen::Matrix<double, 6, 6> xi_j1_ch =
          -0.5 * lgmath::se3::curlyhat(xi_j1);
      const Eigen::Matrix<double, 6, 6> w =
          J_i1 * (omega(1, 0) * J_21_inv +
                  omega(1, 1) * w2_j_21_inv) +
          xi_j1_ch * (omega(0, 0) * J_21_inv +
                      omega(0, 1) * w2_j_21_inv);

      Eigen::Matrix<double, 6, 24> interp_jac_vel =
          Eigen::Matrix<double, 6, 24>::Zero();
      interp_jac_vel.block<6, 6>(0, 0) = -w * Ad_T_21;  // T1
      interp_jac_vel.block<6, 6>(0, 6) =
          (lambda(1, 1) * J_i1 +
           lambda(0, 1) * xi_j1_ch);  // w1
      interp_jac_vel.block<6, 6>(0, 12) = w;      // T2
      interp_jac_vel.block<6, 6>(0, 18) =
          omega(1, 1) * J_i1 * J_21_inv +
          omega(0, 1) * xi_j1_ch * J_21_inv;  // w2
      interp_jac_v1 = interp_jac_vel.block<3, 24>(0, 0);
    }

    const Eigen::Matrix<double, 6, 6> w =
        J_i1 * (omega(0, 0) * J_21_inv +
                omega(0, 1) * w2_j_21_inv);

    // pose interpolation Jacobians
    Eigen::Matrix<double, 6, 24> interp_jac_pose =
        Eigen::Matrix<double, 6, 24>::Zero();

    interp_jac_pose.block<6, 6>(0, 0) = -w * Ad_T_21 + T_i1.adjoint();    // T1
    interp_jac_pose.block<6, 6>(0, 6) = lambda(0, 1) * J_i1;  // w1
    interp_jac_pose.block<6, 6>(0, 12) = w;                               // T2
    interp_jac_pose.block<6, 6>(0, 18) =
        omega(0, 1) * J_i1 * J_21_inv;  // w2

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
    lgmath::se3::Transformation transform_i_to_m = T_mi_1;
    Eigen::Matrix<double, 6, 12> interp_jac_T_m_i =
        Eigen::Matrix<double, 6, 12>::Zero();
    if (transform_i_to_m_1_->active() || transform_i_to_m_2_->active()) {
      const double alpha_ =
          (ts - knot1_->time().seconds()) /
          (knot2_->time().seconds() - knot1_->time().seconds());
      const Eigen::Matrix<double, 6, 1> xi_i1_ =
          alpha_ * (T_mi_2 / T_mi_1).vec();
      transform_i_to_m = lgmath::se3::Transformation(xi_i1_) * T_mi_1;
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
    } else if (transform_i_to_m_1_->active() && !transform_i_to_m_2_->active()) {
      transform_i_to_m = T_mi_1;
      interp_jac_T_m_i.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity();
    } else if (!transform_i_to_m_1_->active() && transform_i_to_m_2_->active()) {
      throw std::runtime_error("either (T_mi_1 and T_mi_2) need to be active or just T_mi_1");
    }

    const Eigen::Matrix3d &C_vm = T_i0.matrix().block<3, 3>(0, 0);
    const Eigen::Matrix3d &C_mi = transform_i_to_m.matrix().block<3, 3>(0, 0);

    double delta_ts = 0.;
    if (i < (int)imu_data_vec_.size() - 1) {
      delta_ts = imu_data_vec_[i + 1].timestamp - imu_data_vec_[i].timestamp;
    } else {
      delta_ts = knot2_->time().seconds() - imu_data_vec_[i].timestamp;
    }
    if (options_.se2) {
      preint_delta_v.block<2, 1>(0, 0) +=
          (imu_data.lin_acc.block<2, 1>(0, 0) - bias_i.block<2, 1>(0, 0)) *
          delta_ts;
    } else {
      preint_delta_v += (imu_data.lin_acc + C_vm * C_mi * options_.gravity -
                         bias_i.block<3, 1>(0, 0)) *
                        delta_ts;
    }

    R += R_acc_ * delta_ts * delta_ts;

    G.block<3, 12>(0, 24) += delta_ts * jac_bias_accel_ * interp_jac_bias;
    if (!options_.se2) {
      G.block<3, 24>(0, 0) +=
          delta_ts * (-1 * lgmath::so3::hat(C_vm * C_mi * options_.gravity)) *
          interp_jac_pose.block<3, 24>(3, 0);
      G.block<3, 12>(0, 36) +=
          delta_ts * (-1 * C_vm * lgmath::so3::hat(C_mi * options_.gravity)) *
          interp_jac_T_m_i.block<3, 12>(3, 0);
    }
  }

  G.block<3, 24>(0, 0) += (-1) * interp_jac_v1;
  G.block<3, 3>(0, 18) += Eigen::Matrix3d::Identity();

  Eigen::Vector3d raw_error = Eigen::Vector3d::Zero();
  if (options_.se2) {
    raw_error.block<2, 1>(0, 0) = w2.block<2, 1>(0, 0) - v1.block<2, 1>(0, 0) +
                                  preint_delta_v.block<2, 1>(0, 0);
    G.block<1, 48>(2, 0).setZero();
  } else {
    raw_error = w2.block<3, 1>(0, 0) - v1 + preint_delta_v;
  }

  StaticNoiseModel<3>::Ptr noise_model = StaticNoiseModel<3>::MakeShared(R);
  const Eigen::Vector3d white_error = noise_model->whitenError(raw_error);
  const double sqrt_w = sqrt(loss_func_->weight(white_error.norm()));
  const Eigen::Vector3d error = sqrt_w * white_error;

  G = sqrt_w * noise_model->getSqrtInformation() * G;
  A = G.transpose() * G;
  b = (-1) * G.transpose() * error;

  std::vector<bool> active;
  active.push_back(knot1_->pose()->active());
  active.push_back(knot1_->velocity()->active());
  active.push_back(knot2_->pose()->active());
  active.push_back(knot2_->velocity()->active());
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
  if (active[3]) {
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
  if (active[4]) {
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
  if (active[5]) {
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
  if (active[6]) {
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
  if (active[7]) {
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

  for (int i = 0; i < 8; ++i) {
    if (!active[i]) continue;
    // Get the key and state range affected
    const auto &key1 = keys[i];
    unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

    // Calculate terms needed to update the right-hand-side
    Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

    // Update the right-hand side (thread critical)

#pragma omp critical(b_update)
    { gradient_vector->mapAt(blkIdx1) += newGradTerm; }

    for (int j = i; j < 8; ++j) {
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
}

}  // namespace steam
