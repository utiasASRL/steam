#include "steam/problem/cost_term/lidar_inertial_marg_cost_term.hpp"

namespace steam {

LidarInertialMarginalizedCostTerm::Ptr
LidarInertialMarginalizedCostTerm::MakeShared(
    const Interface::ConstPtr &interface, const Time &time1,
    const Evaluable<BiasType>::ConstPtr &bias1,
    const Evaluable<PoseType>::ConstPtr &T_mi_1, const Time &time2,
    const Evaluable<BiasType>::ConstPtr &bias2,
    const Evaluable<PoseType>::ConstPtr &T_mi_2, Options options) {
  return std::make_shared<LidarInertialMarginalizedCostTerm>(
      interface, time1, bias1, T_mi_1, time2, bias2, T_mi_2, options);
}

LidarInertialMarginalizedCostTerm::LidarInertialMarginalizedCostTerm(
    const Interface::ConstPtr &interface, const Time &time1,
    const Evaluable<BiasType>::ConstPtr &bias1,
    const Evaluable<PoseType>::ConstPtr &T_mi_1, const Time &time2,
    const Evaluable<BiasType>::ConstPtr &bias2,
    const Evaluable<PoseType>::ConstPtr &T_mi_2, Options options)
    : interface_(interface),
      time1_(time1),
      bias1_(bias1),
      T_mi_1_(T_mi_1),
      time2_(time2),
      bias2_(bias2),
      T_mi_2_(T_mi_2),
      options_(options) {
  knot1_ = interface_.get(time1_);
  knot2_ = interface_.get(time2_);

  acc_loss_func_ = [this]() -> BaseLossFunc::Ptr {
    switch (options_.accel_loss_func) {
      case LOSS_FUNC::L2:
        return L2LossFunc::MakeShared();
      case LOSS_FUNC::DCS:
        return DcsLossFunc::MakeShared(options_.accel_loss_sigma);
      case LOSS_FUNC::CAUCHY:
        return CauchyLossFunc::MakeShared(options_.accel_loss_sigma);
      case LOSS_FUNC::CAUCHY:
        return CauchyLossFunc::MakeShared(options_.accel_loss_sigma);
      default:
        return nullptr;
    }
    return nullptr;
  }();

  const auto gyro_loss_func_ = [this]() -> BaseLossFunc::Ptr {
    switch (options_.gyro_loss_func) {
      case LOSS_FUNC::L2:
        return L2LossFunc::MakeShared();
      case LOSS_FUNC::DCS:
        return DcsLossFunc::MakeShared(options_.gyro_loss_sigma);
      case LOSS_FUNC::CAUCHY:
        return CauchyLossFunc::MakeShared(options_.gyro_loss_sigma);
      case LOSS_FUNC::CAUCHY:
        return CauchyLossFunc::MakeShared(options_.gyro_loss_sigma);
      default:
        return nullptr;
    }
    return nullptr;
  }();

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

  R_acc_.diagonal() = options_.r_imu_acc;
  R_acc_inv_ = R_acc_.inverse();
  acc_noise_model_ = StaticNoiseModel<3>::MakeShared(R_acc_);
  R_ang_.diagonal() = options_.r_imu_ang;
  R_ang_inv_ = R_ang_.inverse();
  gyro_noise_model_ = StaticNoiseModel<3>::MakeShared(R_ang_);
}

/** \brief Compute the cost to the objective function */
double LidarInertialMarginalizedCostTerm::cost() const {/*TODO*/};

/** \brief Get keys of variables related to this cost term */
void LidarInertialMarginalizedCostTerm::getRelatedVarKeys(KeySet &keys) const {
    /*TODO*/};

void LidarInertialMarginalizedCostTerm::addAccelCostTerms(
    const std::vector<AccelerationErrorEvaluator::ConstPtr> &accel_err_vec) {
  accel_err_vec_ = accel_err_vec;

  // accel_cost_terms_.reserve(accel_err_vec.size());
  // for (AccelerationErrorEvaluator::ConstPtr &accel_err : accel_err_vec) {
  //   meas_times_.emplace(accel_err->getTime());
  //   const auto acc_cost = WeightedLeastSqCostTerm<3>::MakeShared(
  //       accel_err, acc_noise_model, acc_loss_func);
  //   accel_cost_terms_.emplace_back(acc_cost);
  // }
}

void LidarInertialMarginalizedCostTerm::addGyroCostTerms(
    const std::vector<GyroErrorEvaluator::ConstPtr> &gyro_err_vec) { /*TODO*/

  gyro_err_vec_ = gyro_err_vec;

  gyro_cost_terms_.reserve(gyro_err_vec.size());
  for (GyroErrorEvaluator::ConstPtr &gyro_err : gyro_err_vec) {
    meas_times_.emplace(gyro_err->getTime());
    const auto gyro_cost = WeightedLeastSqCostTerm<3>::MakeShared(
        gyro_err, gyro_noise_model, gyro_loss_func);
    gyro_cost_terms_.emplace_back(gyro_cost);
  }
}

void LidarInertialMarginalizedCostTerm::addP2PCostTerms(
    const std::vector<P2PErrorEvaluator::ConstPtr> &p2p_err_vec,
    const std::vector<Eigen::Matrix<double, 3, 3>> &W_vec) {
  assert(p2p_err_vec.size() == W_vec.size());
  p2p_err_vec_ = p2p_err_vec;
  p2p_noise_models_.reserve(p2p_err_vec.size());
  for (int i = 0; i < p2p_err_vec.size(); ++i) {
    const P2PErrorEvaluator::ConstPtr &p2p_err = p2p_err_vec[i];
    const Eigen::Matrix<3, 3> &W = W_vec[i];
    meas_times_.emplace(p2p_err->getTime());
    const auto noise_model =
        StaticNoiseModel<3>::MakeShared(W, NoiseType::INFORMATION);
    p2p_noise_models_.emplace_back(noise_model);
  }
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void LidarInertialMarginalizedCostTerm::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  // number of blocks = meas_times_.size() + 1
  std::vector<unsigned int> blk_sizes;

  int end_blks = 4;
  int state_blk_size = 3;
  if (!options_.bias_const_over_window) state_blk_size += 1;
  if (options_.estimate_T_mi) {
    end_blks += 1;
    if (!options_.T_mi_const_over_window) state_blk_size += 1;
  }
  const int num_blks = end_blks * 2 + meas_times_.size() * state_blk_size;
  const int blk_size = 6;
  const int pose_blk_offset = 0;
  const int velocity_blk_offset = 1;
  const int acceleration_blk_offset = 2;
  const int bias_blk_offset = 3;
  const int T_mi_blk_offset = 4;
  const int init_state_index = 0;
  const int end_state_index = end_blks + meas_times_.size() * state_blk_size;

  blk_sizes.reserve(num_blks);
  for (int i = 0; i < num_blks; ++i) {
    blk_sizes.push_back(blk_size);
  }
  BlockSparseMatrix A(blk_sizes, true);
  BlockVector b(blk_sizes);

  std::map<Time, int> time_to_index_map;

  // block (0) corresponds to time1
  // block (n-1) corresponds to time2
  for (int i = end_blks; i < meas_times_.size() * state_blk_size + end_blks;
       i += state_blk_size) {
    time_to_index_map.emplace(meas_times_[i], i);
  }

  // TODO: build prior cost terms

  // TODO: build meas cost terms

  // Accelerometer Error Terms:
  using ErrorType = Eigen::Matrix<double, 3, 1>;
  for (AccelerationErrorEvaluator::ConstPtr &accel_err : accel_err_vec_) {
    ErrorType raw_error = accel_err->value();
    ErrorType white_error = acc_noise_model_->whitenError(raw_error);
    double w = acc_loss_func_->weight(white_error.norm());

    const uint state_index = time_to_index_map.at(accel_err->getTime());

    // todo: create vector of jacobians
    std::vector<std::pair<int, Eigen::MatrixXd>> jacobians;
    int index = state_index;
    if (options_.bias_const_over_window) {
      index = init_state_index + bias_blk_offset;
    } else {
      index = state_index + bias_blk_offset;
    }
    jacobians.emplace_back(std::make_pair(index, accel_err->getJacobianBias()));

    if (options_.estimate_T_mi) {
      if (options_.T_mi_const_over_window) {
        index = init_state_index + T_mi_blk_offset;
      } else {
        index = state_index + T_mi_blk_offset;
      }
    }
    jacobians.emplace_back(std::make_pair(index, accel_err->getJacobianT_mi()));

    // todo: continue adding to the vector of jacobians, copy
    // weightedleastsquares for doing the inner product and getting the hessian
    // and gradient updates

    uint blk_size = blk_sizes[index];
    Eigen::Matrix<double, 3, blk_size> G =
        Eigen::Matrix<double, 3, blk_size>::Zero();
    G.block<3, 6>(0, 0) = accel_err->getJacobianPose();
    G.block<3, 6>(0, 12) = accel_err->getJacobianAcceleration();

    Eigen::Matrix<double, 3, blk_sizes[0]> G0 =
        Eigen::Matrix<double, 3, blk_sizes[0]>::Zero();
    bool use_G0 = false;

    // if (options_.bias_const_over_window) {
    //   G0.block<3, 6>(0, 18) = accel_err->getJacobianBias();
    //   use_G0 = true;
    // } else {
    //   G.block<3, 6>(0, 18) = accel_err->getJacobianBias();
    // }
    if (options_.estimate_T_mi) {
      if (options_.T_mi_const_over_window) {
        G0.block<3, 6>(0, 24) = accel_err->getJacobianT_mi();
        use_G0 = true;
      } else {
        G.block<3, 6>(0, 24) = accel_err->getJacobianT_mi();
      }
    }

    if (use_G0) {
      uint row = 0;
      uint col = 0;
      const Eigen::MatrixXd wGTRinv = w * G0.transpose() * R_acc_inv_;

      // Calculate terms needed to update the right-hand-side
      Eigen::MatrixXd newGradTerm = (-1) * wGTRinv * raw_error;

      // #pragma omp critical(b_update_local)
      { b->mapAt(0) += newGradTerm; }

      const Eigen::MatrixXd newHessianTerm = wGTRinv * G0;
      BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col, true);
      // omp_set_lock(&entry.lock);
      entry.data += newHessianTerm;
      // omp_unset_lock(&entry.lock);
      col = index;
      const Eigen::MatrixXd newHessianTerm2 = wGTRinv * G;
      BlockSparseMatrix::BlockRowEntry &entry2 = A->rowEntryAt(row, col, true);
      // omp_set_lock(&entry2.lock);
      entry2.data += newHessianTerm2;
      // omp_unset_lock(&entry2.lock);
    }

    const uint row = index, col = index;
    const Eigen::MatrixXd wGTRinv = w * G.transpose() * R_acc_inv_;
    Eigen::MatrixXd newGradTerm = (-1) * wGTRinv * raw_error;
    // #pragma omp critical(b_update_local)
    { b->mapAt(index) += newGradTerm; }
    const Eigen::MatrixXd newHessianTerm = wGTRinv * G;
    BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col, true);
    // omp_set_lock(&entry.lock);
    entry.data += newHessianTerm;
    // omp_unset_lock(&entry.lock);
  }

  for (GyroErrorEvaluator::ConstPtr &gyro_err : gyro_err_vec_) {
    ErrorType raw_error = gyro_err->value();
    ErrorType white_error = gyro_noise_model_->whitenError(raw_error);
    double w = gyro_loss_func_->weight(white_error.norm());

    uint index = time_to_index_map.at(gyro_err->getTime());
    uint blk_size = blk_sizes[index];
    Eigen::Matrix<double, 3, blk_size> G =
        Eigen::Matrix<double, 3, blk_size>::Zero();
    G.block<3, 6>(0, 6) = gyro_err->getJacobianVelocity();

    Eigen::Matrix<double, 3, blk_sizes[0]> G0 =
        Eigen::Matrix<double, 3, blk_sizes[0]>::Zero();
    bool use_G0 = false;

    if (options_.bias_const_over_window) {
      G0.block<3, 6>(0, 18) = gyro_err->getJacobianBias();
      use_G0 = true;
    } else {
      G.block<3, 6>(0, 18) = gyro_err->getJacobianBias();
    }

    if (use_G0) {
      uint row = 0;
      uint col = 0;
      const Eigen::MatrixXd wGTRinv = w * G0.transpose() * R_ang_inv_;

      // Calculate terms needed to update the right-hand-side
      Eigen::MatrixXd newGradTerm = (-1) * wGTRinv * raw_error;

      // #pragma omp critical(b_update_local)
      { b->mapAt(0) += newGradTerm; }

      const Eigen::MatrixXd newHessianTerm = wGTRinv * G0;
      BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col, true);
      // omp_set_lock(&entry.lock);
      entry.data += newHessianTerm;
      // omp_unset_lock(&entry.lock);
      col = index;
      const Eigen::MatrixXd newHessianTerm2 = wGTRinv * G;
      BlockSparseMatrix::BlockRowEntry &entry2 = A->rowEntryAt(row, col, true);
      // omp_set_lock(&entry2.lock);
      entry2.data += newHessianTerm2;
      // omp_unset_lock(&entry2.lock);
    }

    const uint row = index, col = index;
    const Eigen::MatrixXd wGTRinv = w * G.transpose() * R_ang_inv_;
    Eigen::MatrixXd newGradTerm = (-1) * wGTRinv * raw_error;
    // #pragma omp critical(b_update_local)
    { b->mapAt(index) += newGradTerm; }
    const Eigen::MatrixXd newHessianTerm = wGTRinv * G;
    BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col, true);
    // omp_set_lock(&entry.lock);
    entry.data += newHessianTerm;
    // omp_unset_lock(&entry.lock);
  }

  for (int i = 0; i < p2p_err_vec.size(); ++i) {
    const P2PErrorEvaluator::ConstPtr &p2p_err = p2p_err_vec_[i];
    const StaticNoiseModel::ConstPtr noise_model = p2p_noise_models_[i];

    ErrorType raw_error = p2p_err->value();
    ErrorType white_error = noise_model->whitenError(raw_error);
    double sqrt_w = sqrt(p2p_loss_func_->weight(white_error.norm()));
    ErrorType error = sqrt_w * white_error;

    uint index = time_to_index_map.at(p2p_err->getTime());
    uint blk_size = blk_sizes[index];
    Eigen::Matrix<double, 3, blk_size> G =
        Eigen::Matrix<double, 3, blk_size>::Zero();
    G.block<3, 6>(0, 0) = p2p_err->getJacobianPose();
    G = sqrt_w * noise_model->getSqrtInformation() * G;

    const uint row = index, col = index;
    Eigen::MatrixXd newGradTerm = (-1) * G.transpose() * error;
    // #pragma omp critical(b_update_local)
    { b->mapAt(index) += newGradTerm; }
    const Eigen::MatrixXd newHessianTerm = G.transpose() * G;
    BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col, true);
    // omp_set_lock(&entry.lock);
    entry.data += newHessianTerm;
    // omp_unset_lock(&entry.lock);
  }
};

}  // namespace steam
