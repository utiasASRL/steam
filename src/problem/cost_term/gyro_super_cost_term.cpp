#include "steam/problem/cost_term/gyro_super_cost_term.hpp"
#include <iostream>

namespace steam {

GyroSuperCostTerm::Ptr GyroSuperCostTerm::MakeShared(
    const Interface::ConstPtr &interface, const Time &time1, const Time &time2,
    const Evaluable<BiasType>::ConstPtr &bias1,
    const Evaluable<BiasType>::ConstPtr &bias2, const Options &options) {
  return std::make_shared<GyroSuperCostTerm>(interface, time1, time2, bias1,
                                             bias2, options);
}

/** \brief Compute the cost to the objective function */
double GyroSuperCostTerm::cost() const {
  double cost = 0;
  using namespace steam::se3;
  using namespace steam::vspace;
  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();
  const auto b1_ = bias1_->forward();
  const auto b2_ = bias2_->forward();

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();
  const auto b1 = b1_->value();
  const auto b2 = b2_->value();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto J_21_inv_w2 = J_21_inv * w2;

#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : cost)
  for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
    const double &ts = imu_data_vec_[i].timestamp;
    const IMUData &imu_data = imu_data_vec_[i];

    // pose interpolation
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv_w2;
    const Eigen::Matrix<double, 6, 1> xi_j1 =
        lambda.block<6, 6>(6, 6) * w1 + omega.block<6, 6>(6, 0) * xi_21 +
        omega.block<6, 6>(6, 6) * J_21_inv_w2;
    // Interpolated velocity
    const Eigen::Matrix<double, 6, 1> w_i = lgmath::se3::vec2jac(xi_i1) * xi_j1;

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

    Eigen::Matrix<double, 3, 1> raw_error_gyro =
        Eigen::Matrix<double, 3, 1>::Zero();

    if (options_.se2) {
      raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
    } else {
      raw_error_gyro =
          imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
    }

    cost += gyro_loss_func_->cost(
        gyro_noise_model_->getWhitenedErrorNorm(raw_error_gyro));
  }
  return cost;
}

/** \brief Get keys of variables related to this cost term */
void GyroSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
  bias1_->getRelatedVarKeys(keys);
  bias2_->getRelatedVarKeys(keys);
}

void GyroSuperCostTerm::init() { initialize_interp_matrices_(); }

void GyroSuperCostTerm::initialize_interp_matrices_() {
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
#pragma omp parallel for num_threads(options_.num_threads)
  for (const IMUData &imu_data : imu_data_vec_) {
    const double &time = imu_data.timestamp;
    // if (interp_mats_.find(time) == interp_mats_.end()) {
    // Get Lambda, Omega for this time
    const double tau = time - time1_.seconds();
    const double kappa = knot2_->time().seconds() - time;
    const Matrix12d Q_tau = steam::traj::const_vel::getQ(tau, ones);
    const Matrix12d Tran_kappa = steam::traj::const_vel::getTran(kappa);
    const Matrix12d Tran_tau = steam::traj::const_vel::getTran(tau);
    const Matrix12d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
    const Matrix12d lambda = (Tran_tau - omega * Tran_T_);
    const auto omega_lambda = std::make_pair(omega, lambda);
#pragma omp critical
    interp_mats_.emplace(time, omega_lambda);
    // }
  }
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void GyroSuperCostTerm::buildGaussNewtonTerms(
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

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();
  const auto b1 = b1_->value();
  const auto b2 = b2_->value();

  Eigen::Matrix<double, 36, 36> A = Eigen::Matrix<double, 36, 36>::Zero();
  Eigen::Matrix<double, 36, 1> b = Eigen::Matrix<double, 36, 1>::Zero();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const auto Ad_T_21 = lgmath::se3::tranAd(T_21.matrix());
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto w2_j_21_inv = 0.5 * lgmath::se3::curlyhat(w2) * J_21_inv;
  const auto J_21_inv_w2 = J_21_inv * w2;

  // If some variables are not active? (simply don't use those parts
  // of the A, b to update hessian, grad at the end)
#pragma omp declare reduction(+ : Eigen::Matrix<double, 36, 36> : omp_out = \
                                  omp_out + omp_in)                         \
    initializer(omp_priv = Eigen::Matrix<double, 36, 36>::Zero())
#pragma omp declare reduction(+ : Eigen::Matrix<double, 36, 1> : omp_out = \
                                  omp_out + omp_in)                        \
    initializer(omp_priv = Eigen::Matrix<double, 36, 1>::Zero())
#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : A) \
    reduction(+ : b)
  for (int i = 0; i < (int)imu_data_vec_.size(); ++i) {
    const double &ts = imu_data_vec_[i].timestamp;
    const IMUData &imu_data = imu_data_vec_[i];

    // interpolation
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv_w2;
    const Eigen::Matrix<double, 6, 1> xi_j1 =
        lambda.block<6, 6>(6, 6) * w1 + omega.block<6, 6>(6, 0) * xi_21 +
        omega.block<6, 6>(6, 6) * J_21_inv_w2;
    // Interpolated velocity
    const Eigen::Matrix<double, 6, 1> w_i = lgmath::se3::vec2jac(xi_i1) * xi_j1;

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

    // velocity interpolation Jacobians
    Eigen::Matrix<double, 6, 24> interp_jac_vel =
        Eigen::Matrix<double, 6, 24>::Zero();

    const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);

    const Eigen::Matrix<double, 6, 6> xi_j1_ch =
        -0.5 * lgmath::se3::curlyhat(xi_j1);

    Eigen::Matrix<double, 6, 6> w =
        J_i1 * (omega.block<6, 6>(6, 0) * J_21_inv +
                omega.block<6, 6>(6, 6) * w2_j_21_inv) +
        xi_j1_ch * (omega.block<6, 6>(0, 0) * J_21_inv +
                    omega.block<6, 6>(0, 6) * w2_j_21_inv);

    interp_jac_vel.block<6, 6>(0, 0) = -w * Ad_T_21;  // T1
    interp_jac_vel.block<6, 6>(0, 6) =
        (lambda.block<6, 6>(6, 6) * J_i1 +
         lambda.block<6, 6>(0, 6) * xi_j1_ch);  // w1
    interp_jac_vel.block<6, 6>(0, 12) = w;      // T2
    interp_jac_vel.block<6, 6>(0, 18) =
        omega.block<6, 6>(6, 6) * J_i1 * J_21_inv +
        omega.block<6, 6>(0, 6) * xi_j1_ch * J_21_inv;  // w2

    // evaluate, weight, whiten error
    Eigen::Matrix<double, 3, 1> raw_error_gyro =
        Eigen::Matrix<double, 3, 1>::Zero();

    if (options_.se2) {
      raw_error_gyro(2, 0) = imu_data.ang_vel(2, 0) + w_i(5, 0) - bias_i(5, 0);
    } else {
      raw_error_gyro =
          imu_data.ang_vel + w_i.block<3, 1>(3, 0) - bias_i.block<3, 1>(3, 0);
    }
    const Eigen::Matrix<double, 3, 1> white_error_gyro =
        gyro_noise_model_->whitenError(raw_error_gyro);
    const double sqrt_w_gyro =
        sqrt(gyro_loss_func_->weight(white_error_gyro.norm()));
    const Eigen::Matrix<double, 3, 1> error_gyro =
        sqrt_w_gyro * white_error_gyro;

    // Get Jacobians
    Eigen::Matrix<double, 3, 36> G = Eigen::Matrix<double, 3, 36>::Zero();
    G.block<3, 24>(0, 0) = sqrt_w_gyro *
                           gyro_noise_model_->getSqrtInformation() * jac_vel_ *
                           interp_jac_vel;
    G.block<3, 12>(0, 24) = sqrt_w_gyro *
                            gyro_noise_model_->getSqrtInformation() *
                            jac_bias_ * interp_jac_bias;

    A += G.transpose() * G;
    b += (-1) * G.transpose() * error_gyro;
  }

  std::vector<bool> active;
  active.push_back(knot1_->pose()->active());
  active.push_back(knot1_->velocity()->active());
  active.push_back(knot2_->pose()->active());
  active.push_back(knot2_->velocity()->active());
  active.push_back(bias1_->active());
  active.push_back(bias2_->active());

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
  // std::cout << "keys:";
  // for (unsigned int i = 0; i < keys.size(); ++i) {
  //   if (!active[i]) continue;
  //   std::cout << state_vec.getStateBlockIndex(keys[i]) << " ";
  // }
  // std::cout << std::endl;

  for (int i = 0; i < 6; ++i) {
    if (!active[i]) continue;
    // Get the key and state range affected
    const auto &key1 = keys[i];
    unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

    // Calculate terms needed to update the right-hand-side
    Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

    // Update the right-hand side (thread critical)

#pragma omp critical(b_update)
    { gradient_vector->mapAt(blkIdx1) += newGradTerm; }

    for (int j = i; j < 6; ++j) {
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
