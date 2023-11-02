#include "steam/problem/cost_term/lidar_inertial_interp_cost_term.hpp"

namespace steam {

P2PSuperCostTerm::Ptr P2PSuperCostTerm::MakeShared(
    const Interface::ConstPtr &interface, const Time &time1, const Time &time2,
    Options options) {
  return std::make_shared<P2PSuperCostTerm>(interface, time1, time2, options);
}

P2PSuperCostTerm::P2PSuperCostTerm(const Interface::ConstPtr &interface,
                                   const Time &time1, const Time &time2,
                                   Options options)
    : interface_(interface), time1_(time1), time2_(time2), options_(options) {
  knot1_ = interface_.get(time1_);
  knot2_ = interface_.get(time2_);

  const double T = (knot2->time() - knot1->time()).seconds();
  Matrix18d Qinv_T_ = interface_->getQinvPublic(T, ones);
  Matrix18d Tran_T_ = interface_->getTranPublic(T);

  const auto p2p_loss_func_ = [this]() -> BaseLossFunc::Ptr {
    switch (options_.p2p_loss_func) {
      case LOSS_FUNC::L2:
        return L2LossFunc::MakeShared();
      case LOSS_FUNC::DCS:
        return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
      case LOSS_FUNC::CAUCHY:
        return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
      case LOSS_FUNC::GM:
        return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
      default:
        return nullptr;
    }
    return nullptr;
  }();
}

/** \brief Compute the cost to the objective function */
double P2PSuperCostTerm::cost() const {/*TODO*/};

/** \brief Get keys of variables related to this cost term */
void P2PSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {/*TODO*/};

void P2PSuperCostTerm::setP2PMatches(const std::vector<P2PMatch> &p2p_matches) {
  p2p_matches_ = p2p_matches;
  meas_times_.clear();
  for (const auto &p2p_match : p2p_matches) {
    meas_times_.insert(p2p_match.timestamp);
  }
  initialize_interp_matrices_();
}

void P2PSuperCostTerm::initialize_interp_matrices_() {
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
  for (auto &time : meas_times_) {
    if (interp_mats_.find(time) == inter_mats_.end()) {
      // Get Lambda, Omega for this time
      const double tau = (time - time1_).seconds();
      const double kappa = (time2_ - time).seconds();
      const Matrix18d Q_tau = interface_->getQPublic(tau, ones);
      const Matrix18d Tran_kappa = interface_->getTranPublic(kappa);
      const Matrix18d Tran_tau = interface_->getTranPublic(tau);
      const Matrix18d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
      const Matrix18d lambda = (Tran_tau - omega * Tran_T_);
      interp_mats_.emplace(time, std::make_pair(omega, lambda));
    }
  }
}

// See State Estimation (2nd Ed) Section 11.1.4 and page 66 of Tim Tang's
thesis void P2PSuperCostTerm::getMotionPriorJacobians_(
    const lgmath::se3::Transformation &T1,
    const lgmath::se3::Transformation &T2,
    const Eigen::Matrix<double, 6, 1> &w2,
    const Eigen::Matrix<double, 6, 1> &dw2,
    const Eigen::Matrix<double, 18, 18> &Phi, Eigen::Matrix<double, 18, 18> &F,
    Eigen::Matrix<double, 18, 18> &E) const {
  const auto T_21 = T2 / T1;
  const auto xi_21 = T_21.vec();
  const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto Jinv_12 = J_21_inv * T_21.adjoint();

  F.setZero();
  // pose
  F.block<6, 6>(0, 0) = -Jinv_12;
  F.block<6, 6>(6, 0) = -0.5 * lgmath::se3::curlyhat(w2) * Jinv_12;
  F.block<6, 6>(12, 0) =
      -0.25 * lgmath::se3::curlyhat(w2) * lgmath::se3::curlyhat(w2) * Jinv_12 -
      0.5 * lgmath::se3::curlyhat(dw2) * Jinv_12;
  // velocity
  F.block<6, 6>(0, 6) = -Phi.block<6, 6>(0, 6);
  F.block<6, 6>(6, 6) = -Phi.block<6, 6>(6, 6);
  F.block<6, 6>(12, 6) = Eigen::Matrix<double, 6, 6>::Zero();
  // acceleration
  F.block<6, 6>(0, 12) = -Phi.block<6, 6>(0, 12);
  F.block<6, 6>(6, 12) = -Phi.block<6, 6>(6, 12);
  F.block<6, 6>(12, 12) = -Phi.block<6, 6>(12, 12);

  E.setZero();
  // pose
  E.block<6, 6>(0, 0) = J_21_inv;
  E.block<6, 6>(6, 0) = 0.5 * lgmath::se3::curlyhat(w2) * J_21_inv;
  E.block<6, 6>(12, 0) =
      0.25 * lgmath::se3::curlyhat(w2) * lgmath::se3::curlyhat(w2) * J_21_inv +
      0.5 * lgmath::se3::curlyhat(dw2) * J_21_inv;
  // velocity
  E.block<6, 6>(6, 6) = J_21_inv;
  E.block<6, 6>(12, 6) = -0.5 * lgmath::se3::curlyhat(J_21_inv * w2) +
                         0.5 * lgmath::se3::curlyhat(w2) * J_21_inv;
  // acceleration
  E.block<6, 6>(12, 12) = J_21_inv;
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void P2PSuperCostTerm::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  // todo: get block indices to update approximate_hessian and gradient_vector
  const int blk_size = 6;
  const int state_blk_size = 3;
  const int num_blks = state_blk_size * 2;
  std::vector<int> blk_sizes;
  for (int i = 0; i < num_blks; ++i) {
    blk_sizes.push_back(blk_size);
  }
  BlockSparseMatrix A(blk_sizes, true);
  BlockVector b(blk_sizes);

  const auto T1 = knot1_->pose()->value();
  const auto w1 = knot1_->velocity()->value();
  const auto dw1 = knot1_->acceleration()->value();
  const auto T2 = knot2_->pose()->value();
  const auto w2 = knot2_->velocity()->value();
  const auto dw2 = knot2_->acceleration()->value();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto J_21_inv_w2 = J_21_inv * w2;
  const auto J_21_inv_curl_dw2 =
      (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

  std::map<double, Eigen::Matrix4d> T_ms_cache;
  std::map<double, std::vector<Matrix6d>> interp_jacobian_cache;

  // interpolate pose and get interpolation jacobians at only the unique
  // measurement times
#pragma omp parallel for num_threads(options_.num_threads)
  for (const auto &ts : meas_times_) {
    // pose interpolation
    const auto &omega_lambda = interp_mats_[ts];
    const auto &omega = omega_lambda.first;
    const auto &lambda = omega_lambda.second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
        omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv_w2 +
        omega.block<6, 6>(0, 12) * J_21_inv_curl_dw2;
    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;
    const Eigen::Matrix4d T_ms = (options_.T_sr * T_i0).inverse().matrix();
#pragma omp critical { T_ms_cache[ts] = T_ms; }
    // pose interpolation Jacobians
    std::vector<Matrix6d> jacs;
    const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);

    const Eigen::Matrix<double, 6, 6> w =
        J_i1 *
        (omega.block<6, 6>(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() +
         omega.block<6, 6>(0, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
         omega.block<6, 6>(0, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
             lgmath::se3::curlyhat(w2) +
         omega.block<6, 6>(0, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
        J_21_inv;

    jacs.push_back(-w * T_21.adjoint() + T_i1.adjoint());  // T1
    jacs.push_back(lambda.block<6, 6>(0, 6) * J_i1);       // w1
    jacs.push_back(lambda.block<6, 6>(0, 12) * J_i1);      // dw1
    jacs.push_back(w);                                     // T2
    jacs.push_back(omega.block<6, 6>(0, 6) * J_i1 * J_21_inv +
                   omega.block<6, 6>(0, 12) * -0.5 * J_i1 *
                       (lgmath::se3::curlyhat(J_21_inv * w2) -
                        lgmath::se3::curlyhat(w2) * J_21_inv));  // w2
    jacs.push_back(omega.block<6, 6>(0, 12) * J_i1 * J_21_inv);  // dw2
#pragma omp critical { interp_jacobian_cache[ts] = jacs; }
  }

  // todo: what if some variables are not active? (simply don't use those parts
  // of the A, b to update hessian, grad at the end)

  // todo: use a parallel for with a reduction on A and b instead of critical
  // todo: you can just use an Eigen matrix for local A, b since they
  // are small and dense
  // todo: sort p2p_matches into bins with equal timestamps ...
  // for each of these bins, do the A, b update ...
  // no need for the above caches if you do it this way.
  // create std::vector<std::vector<P2PMatch>> ?

#pragma omp parallel for num_threads(options_.num_threads)
  for (const auto &p2p_match : p2p_matches_) {
    // get measurement Jacobian

    // get interpolation Jacobians

    // todo: remember T_sr compose and inverse() jacobians...

    // get interpolation Jacobian (cache)

    // update local (A,b) with A += G.T * G and b += G.T * err
  }

  const double T = (knot2->time() - knot1->time()).seconds();
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
  const Matrix18d Qinv_T = interface_->getQinvPublic(T, ones);
  const Matrix18d Tran_T = interface_->getTranPublic(T);

  lgmath::se3::Transformation T_i0_prev = T1;
  Eigen::Matrix<double, 6, 1> w_i_prev = w1;
  Eigen::Matrix<double, 6, 1> dw_i_prev = dw1;
  auto it = time_to_index_map.begin();
  it++;
  for (; it != time_to_index_map.end(); ++it) {
    const Time &time = it->first;
    const uint state_index = it->second;

    // Get Lambda, Omega for this time
    const double tau = (time - time1_).seconds();
    const double kappa = (time2_ - time).seconds();
    const Matrix18d Q_tau = interface_->getQPublic(tau, ones);
    const Matrix18d Tran_kappa = interface_->getTranPublic(kappa);
    const Matrix18d Tran_tau = interface_->getTranPublic(tau);
    const Matrix18d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T);
    const Matrix18d lambda = (Tran_tau - omega * Tran_T);

    // Calculate interpolated relative se3 algebra
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
        omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv * w2 +
        omega.block<6, 6>(0, 12) *
            (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);
    const Eigen::Matrix<double, 6, 1> xi_j1 =
        lambda.block<6, 6>(6, 6) * w1 + lambda.block<6, 6>(6, 12) * dw1 +
        omega.block<6, 6>(6, 0) * xi_21 +
        omega.block<6, 6>(6, 6) * J_21_inv * w2 +
        omega.block<6, 6>(6, 12) *
            (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);
    const Eigen::Matrix<double, 6, 1> xi_k1 =
        lambda.block<6, 6>(12, 6) * w1 + lambda.block<6, 6>(12, 12) * dw1 +
        omega.block<6, 6>(12, 0) * xi_21 +
        omega.block<6, 6>(12, 6) * J_21_inv * w2 +
        omega.block<6, 6>(12, 12) *
            (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;  // interpolated pose
    const auto w_i =
        lgmath::se3::vec2jac(xi_i1) * xi_j1;  // interpolated velocity
    const auto dw_i = lgmath::se3::vec2jac(xi_i1) *
                      (xi_k1 + 0.5 * lgmath::se3::curlyhat(xi_j1) *
                                   w_i);  // interpolated acceleration

    // Get motion prior Jacobians
    Matrix18d Phi = interface_->getTranPublic((time - tprev).seconds());
    Matrix18d F = Matrix18d::Zero();
    Matrix18d E = Matrix18d::Zero();
    getMotionPriorJacobians_(T_i0_prev, T_i0, w_i, dw_i, Phi, F, E);

    //
    tprev = time;
    T_i0_prev = T_i0;
    w_i_prev = w_i;
    dw_i_prev = dw_i;
  }

  // Accelerometer Error Terms:
  using ErrorType = Eigen::Matrix<double, 3, 1>;
  using JacType = Eigen::Matrix<double, 3, 6>;
  for (AccelerationErrorEvaluator::ConstPtr &accel_err : accel_err_vec_) {
    const ErrorType raw_error = accel_err->value();
    const ErrorType white_error = acc_noise_model_->whitenError(raw_error);
    const double sqrt_w = sqrt(acc_loss_func_->weight(white_error.norm()));
    const ErrorType error = sqrt_w * white_error;

    const uint state_index = time_to_index_map.at(accel_err->getTime());
    JacType G_pose, G_accel, G_bias, G_T_mi;
    G_pose.setZero();
    G_accel.setZero();
    G_bias.setZero();
    G_T_mi.setZero();
    accel_err->getMeasJacobians(G_pos, G_accel, G_bias, G_T_mi);
    // weight and whiten Jacobians
    G_pos = sqrt_w * acc_noise_model_->getSqrtInformation * G_pos;
    G_accel = sqrt_w * acc_noise_model_->getSqrtInformation * G_accel;
    G_bias = sqrt_w * acc_noise_model_->getSqrtInformation * G_bias;
    G_T_mi = sqrt_w * acc_noise_model_->getSqrtInformation * G_T_mi;

    std::map<int, JacType> jacobians;

    jacobians[state_index + pose_blk_offset] = G_pose;
    jacobians[state_index + acceleration_blk_offset] = G_accel;

    int index = state_index;
    if (options_.bias_const_over_window) {
      index = init_state_index + bias_blk_offset;
    } else {
      index = state_index + bias_blk_offset;
    }
    jacobians[index] = G_bias;

    if (options_.estimate_T_mi) {
      if (options_.T_mi_const_over_window) {
        index = init_state_index + T_mi_blk_offset;
      } else {
        index = state_index + T_mi_blk_offset;
      }
      jacobians[index] = G_T_mi;
    }

    for (auto it = jacobians.begin(); it != jacobians.end(); ++it) {
      const uint blkIdx1 = it->first;
      const auto &jac1 = it->second;
      const Eigen::MatrixXd newGradTerm = (-1) * jac1.transpose() * error;
      { b->mapAt(blkIdx1) += newGradTerm; }

      for (auto it2 = it; it2 != jacobians.end(); ++it2) {
        const uint blkIdx2 = it2->first;
        const auto &jac2 = it2->second;
        uint row, col;
        const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
          if (blkIdx1 <= blkIdx2) {
            row = blkIdx1;
            col = blkIdx2;
            return jac1.transpose() * jac2;
          } else {
            row = blkIdx2;
            col = blkIdx1;
            return jac2.transpose() * jac1;
          }
        }();
        BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col, true);
        entry.data += newHessianTerm;
      }
    }
  }

  for (GyroErrorEvaluator::ConstPtr &gyro_err : gyro_err_vec_) {
    const ErrorType raw_error = gyro_err->value();
    const ErrorType white_error = gyro_noise_model_->whitenError(raw_error);
    const double sqrt_w = sqrt(gyro_loss_func_->weight(white_error.norm()));
    const ErrorType error = sqrt_w * white_error;

    const uint state_index = time_to_index_map.at(gyro_err->getTime());
    JacType G_vel, G_bias;
    G_vel.setZero();
    G_bias.setZero();
    gyro_err->getMeasJacobians(G_vel, G_bias);
    G_vel = sqrt_w * gyro_noise_model_->getSqrtInformation * G_vel;
    G_vias = sqrt_w * gyro_noise_model_->getSqrtInformation * G_bias;

    std::map<int, JacType> jacobians;

    jacobians[state_index + velocity_blk_offset] = G_vel;

    int index = state_index;
    if (options_.bias_const_over_window) {
      index = init_state_index + bias_blk_offset;
    } else {
      index = state_index + bias_blk_offset;
    }
    jacobians[index] = G_bias;

    for (auto it = jacobians.begin(); it != jacobians.end(); ++it) {
      const uint blkIdx1 = it->first;
      const auto &jac1 = it->second;
      const Eigen::MatrixXd newGradTerm = (-1) * jac1.transpose() * error;
      { b->mapAt(blkIdx1) += newGradTerm; }

      for (auto it2 = it; it2 != jacobians.end(); ++it2) {
        const uint blkIdx2 = it2->first;
        const auto &jac2 = it2->second;
        uint row, col;
        const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
          if (blkIdx1 <= blkIdx2) {
            row = blkIdx1;
            col = blkIdx2;
            return jac1.transpose() * jac2;
          } else {
            row = blkIdx2;
            col = blkIdx1;
            return jac2.transpose() * jac1;
          }
        }();
        BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col, true);
        entry.data += newHessianTerm;
      }
    }
  }

  // todo: create map of previously computed Jacobians (speedup)
  std::map<Time, JacType> p2p_jac_map;
  for (int i = 0; i < p2p_err_vec.size(); ++i) {
    const P2PErrorEvaluator::ConstPtr &p2p_err = p2p_err_vec_[i];
    const StaticNoiseModel::ConstPtr noise_model = p2p_noise_models_[i];

    const ErrorType raw_error = p2p_err->value();
    const ErrorType white_error = noise_model->whitenError(raw_error);
    const double sqrt_w = sqrt(p2p_loss_func_->weight(white_error.norm()));
    const ErrorType error = sqrt_w * white_error;

    const uint index = time_to_index_map.at(p2p_err->getTime());
    JacType G;
    if (auto search = p2p_jac_map.find(p2p_err->getTime());
        search != p2p_jac_map.end()) {
      G = search->second;
    } else {
      G = p2p_err->getJacobianPose();
      p2p_jac_map[p2p_err->getTime()] = G;
    }

    const JacType G2 = sqrt_w * noise_model->getSqrtInformation() * G;

    const Eigen::MatrixXd newGradTerm = (-1) * G2.transpose() * error;
    { b->mapAt(index) += newGradTerm; }
    const Eigen::MatrixXd newHessianTerm = G2.transpose() * G2;
    BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(index, index, true);
    entry.data += newHessianTerm;
  }
};

}  // namespace steam
