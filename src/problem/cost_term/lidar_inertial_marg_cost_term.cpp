// #include "steam/problem/cost_term/lidar_inertial_marg_cost_term.hpp"

// namespace steam {

// LidarInertialMarginalizedCostTerm::Ptr
// LidarInertialMarginalizedCostTerm::MakeShared(
//     const Interface::ConstPtr &interface, const Time &time1,
//     const Evaluable<BiasType>::ConstPtr &bias1,
//     const Evaluable<PoseType>::ConstPtr &T_mi_1, const Time &time2,
//     const Evaluable<BiasType>::ConstPtr &bias2,
//     const Evaluable<PoseType>::ConstPtr &T_mi_2, Options options) {
//   return std::make_shared<LidarInertialMarginalizedCostTerm>(
//       interface, time1, bias1, T_mi_1, time2, bias2, T_mi_2, options);
// }

// LidarInertialMarginalizedCostTerm::LidarInertialMarginalizedCostTerm(
//     const Interface::ConstPtr &interface, const Time &time1,
//     const Evaluable<BiasType>::ConstPtr &bias1,
//     const Evaluable<PoseType>::ConstPtr &T_mi_1, const Time &time2,
//     const Evaluable<BiasType>::ConstPtr &bias2,
//     const Evaluable<PoseType>::ConstPtr &T_mi_2, Options options)
//     : interface_(interface),
//       time1_(time1),
//       bias1_(bias1),
//       T_mi_1_(T_mi_1),
//       time2_(time2),
//       bias2_(bias2),
//       T_mi_2_(T_mi_2),
//       options_(options) {
//   knot1_ = interface_.get(time1_);
//   knot2_ = interface_.get(time2_);

//   acc_loss_func_ = [this]() -> BaseLossFunc::Ptr {
//     switch (options_.accel_loss_func) {
//       case LOSS_FUNC::L2:
//         return L2LossFunc::MakeShared();
//       case LOSS_FUNC::DCS:
//         return DcsLossFunc::MakeShared(options_.accel_loss_sigma);
//       case LOSS_FUNC::CAUCHY:
//         return CauchyLossFunc::MakeShared(options_.accel_loss_sigma);
//       case LOSS_FUNC::CAUCHY:
//         return CauchyLossFunc::MakeShared(options_.accel_loss_sigma);
//       default:
//         return nullptr;
//     }
//     return nullptr;
//   }();

//   const auto gyro_loss_func_ = [this]() -> BaseLossFunc::Ptr {
//     switch (options_.gyro_loss_func) {
//       case LOSS_FUNC::L2:
//         return L2LossFunc::MakeShared();
//       case LOSS_FUNC::DCS:
//         return DcsLossFunc::MakeShared(options_.gyro_loss_sigma);
//       case LOSS_FUNC::CAUCHY:
//         return CauchyLossFunc::MakeShared(options_.gyro_loss_sigma);
//       case LOSS_FUNC::CAUCHY:
//         return CauchyLossFunc::MakeShared(options_.gyro_loss_sigma);
//       default:
//         return nullptr;
//     }
//     return nullptr;
//   }();

//   const auto p2p_loss_func_ = [this]() -> BaseLossFunc::Ptr {
//     switch (options_.p2p_loss_func) {
//       case LOSS_FUNC::L2:
//         return L2LossFunc::MakeShared();
//       case LOSS_FUNC::DCS:
//         return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
//       case LOSS_FUNC::CAUCHY:
//         return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
//       case LOSS_FUNC::GM:
//         return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
//       default:
//         return nullptr;
//     }
//     return nullptr;
//   }();

//   R_acc_.diagonal() = options_.r_imu_acc;
//   R_acc_inv_ = R_acc_.inverse();
//   acc_noise_model_ = StaticNoiseModel<3>::MakeShared(R_acc_);
//   R_ang_.diagonal() = options_.r_imu_ang;
//   R_ang_inv_ = R_ang_.inverse();
//   gyro_noise_model_ = StaticNoiseModel<3>::MakeShared(R_ang_);
// }

// /** \brief Compute the cost to the objective function */
// double LidarInertialMarginalizedCostTerm::cost() const {/*TODO*/};

// /** \brief Get keys of variables related to this cost term */
// void LidarInertialMarginalizedCostTerm::getRelatedVarKeys(KeySet &keys) const
// {
//     /*TODO*/};

// void LidarInertialMarginalizedCostTerm::addAccelCostTerms(
//     const std::vector<AccelerationErrorEvaluator::ConstPtr> &accel_err_vec) {
//   accel_err_vec_ = accel_err_vec;

//   // accel_cost_terms_.reserve(accel_err_vec.size());
//   // for (AccelerationErrorEvaluator::ConstPtr &accel_err : accel_err_vec) {
//   //   meas_times_.emplace(accel_err->getTime());
//   //   const auto acc_cost = WeightedLeastSqCostTerm<3>::MakeShared(
//   //       accel_err, acc_noise_model, acc_loss_func);
//   //   accel_cost_terms_.emplace_back(acc_cost);
//   // }
// }

// void LidarInertialMarginalizedCostTerm::addGyroCostTerms(
//     const std::vector<GyroErrorEvaluator::ConstPtr> &gyro_err_vec) { /*TODO*/

//   gyro_err_vec_ = gyro_err_vec;

//   gyro_cost_terms_.reserve(gyro_err_vec.size());
//   for (GyroErrorEvaluator::ConstPtr &gyro_err : gyro_err_vec) {
//     meas_times_.emplace(gyro_err->getTime());
//     const auto gyro_cost = WeightedLeastSqCostTerm<3>::MakeShared(
//         gyro_err, gyro_noise_model, gyro_loss_func);
//     gyro_cost_terms_.emplace_back(gyro_cost);
//   }
// }

// void LidarInertialMarginalizedCostTerm::addP2PCostTerms(
//     const std::vector<P2PErrorEvaluator::ConstPtr> &p2p_err_vec,
//     const std::vector<Eigen::Matrix<double, 3, 3>> &W_vec) {
//   assert(p2p_err_vec.size() == W_vec.size());
//   p2p_err_vec_ = p2p_err_vec;
//   p2p_noise_models_.reserve(p2p_err_vec.size());
//   for (int i = 0; i < p2p_err_vec.size(); ++i) {
//     const P2PErrorEvaluator::ConstPtr &p2p_err = p2p_err_vec[i];
//     const Eigen::Matrix<3, 3> &W = W_vec[i];
//     meas_times_.emplace(p2p_err->getTime());
//     const auto noise_model =
//         StaticNoiseModel<3>::MakeShared(W, NoiseType::INFORMATION);
//     p2p_noise_models_.emplace_back(noise_model);
//   }
// }

// // See State Estimation (2nd Ed) Section 11.1.4 and page 66 of Tim Tang's
// thesis void LidarInertialMarginalizedCostTerm::getMotionPriorJacobians_(
//     const lgmath::se3::Transformation &T1,
//     const lgmath::se3::Transformation &T2,
//     const Eigen::Matrix<double, 6, 1> &w2,
//     const Eigen::Matrix<double, 6, 1> &dw2,
//     const Eigen::Matrix<double, 18, 18> &Phi, Eigen::Matrix<double, 18, 18>
//     &F, Eigen::Matrix<double, 18, 18> &E) const {
//   const auto T_21 = T2 / T1;
//   const auto xi_21 = T_21.vec();
//   const auto J_21_inv = lgmath::se3::vec2jacinv(xi_21);
//   const auto Jinv_12 = J_21_inv * T_21.adjoint();

//   F.setZero();
//   // pose
//   F.block<6, 6>(0, 0) = -Jinv_12;
//   F.block<6, 6>(6, 0) = -0.5 * lgmath::se3::curlyhat(w2) * Jinv_12;
//   F.block<6, 6>(12, 0) =
//       -0.25 * lgmath::se3::curlyhat(w2) * lgmath::se3::curlyhat(w2) * Jinv_12
//       - 0.5 * lgmath::se3::curlyhat(dw2) * Jinv_12;
//   // velocity
//   F.block<6, 6>(0, 6) = -Phi.block<6, 6>(0, 6);
//   F.block<6, 6>(6, 6) = -Phi.block<6, 6>(6, 6);
//   F.block<6, 6>(12, 6) = Eigen::Matrix<double, 6, 6>::Zero();
//   // acceleration
//   F.block<6, 6>(0, 12) = -Phi.block<6, 6>(0, 12);
//   F.block<6, 6>(6, 12) = -Phi.block<6, 6>(6, 12);
//   F.block<6, 6>(12, 12) = -Phi.block<6, 6>(12, 12);

//   E.setZero();
//   // pose
//   E.block<6, 6>(0, 0) = J_21_inv;
//   E.block<6, 6>(6, 0) = 0.5 * lgmath::se3::curlyhat(w2) * J_21_inv;
//   E.block<6, 6>(12, 0) =
//       0.25 * lgmath::se3::curlyhat(w2) * lgmath::se3::curlyhat(w2) * J_21_inv
//       + 0.5 * lgmath::se3::curlyhat(dw2) * J_21_inv;
//   // velocity
//   E.block<6, 6>(6, 6) = J_21_inv;
//   E.block<6, 6>(12, 6) = -0.5 * lgmath::se3::curlyhat(J_21_inv * w2) +
//                          0.5 * lgmath::se3::curlyhat(w2) * J_21_inv;
//   // acceleration
//   E.block<6, 6>(12, 12) = J_21_inv;
// }

// /**
//  * \brief Add the contribution of this cost term to the left-hand (Hessian)
//  * and right-hand (gradient vector) sides of the Gauss-Newton system of
//  * equations.
//  */
// void LidarInertialMarginalizedCostTerm::buildGaussNewtonTerms(
//     const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
//     BlockVector *gradient_vector) const {
//   // number of blocks = meas_times_.size() + 1
//   std::vector<unsigned int> blk_sizes;

//   int end_blks = 4;
//   int state_blk_size = 3;
//   if (!options_.bias_const_over_window) state_blk_size += 1;
//   if (options_.estimate_T_mi) {
//     end_blks += 1;
//     if (!options_.T_mi_const_over_window) state_blk_size += 1;
//   }
//   const int num_blks = end_blks * 2 + meas_times_.size() * state_blk_size;
//   const int blk_size = 6;
//   const int pose_blk_offset = 0;
//   const int velocity_blk_offset = 1;
//   const int acceleration_blk_offset = 2;
//   const int bias_blk_offset = 3;
//   const int T_mi_blk_offset = 4;
//   const int init_state_index = 0;
//   const int end_state_index = end_blks + meas_times_.size() * state_blk_size;

//   blk_sizes.reserve(num_blks);
//   for (int i = 0; i < num_blks; ++i) {
//     blk_sizes.push_back(blk_size);
//   }
//   BlockSparseMatrix A(blk_sizes, true);
//   BlockVector b(blk_sizes);

//   std::map<Time, int> time_to_index_map;

//   // block (0) corresponds to time1
//   // block (n-1) corresponds to time2
//   for (int i = end_blks; i < meas_times_.size() * state_blk_size + end_blks;
//        i += state_blk_size) {
//     time_to_index_map.emplace(meas_times_[i], i);
//   }

//   // TODO: build prior cost terms
//   time_to_index_map.emplace(time1_, init_state_index);
//   time_to_index_map.emplace(time2_, end_state_index);

//   // todo: build map of <Time, Q_inv> to save computation time

//   using Matrix18d = Eigen::Matrix<double, 18, 18>;
//   std::map<Time, Matrix18d> time_to_Q_inv_map;

//   const auto T1 = knot1_->pose()->value();
//   const auto w1 = knot1_->velocity()->value();
//   const auto dw1 = knot1_->acceleration()->value();
//   const auto T2 = knot2_->pose()->value();
//   const auto w2 = knot2_->velocity()->value();
//   const auto dw2 = knot2_->acceleration()->value();

//   const auto xi_21 = (T2 / T1).vec();
//   const Eigen::Matrix<double, 6, 6> J_21_inv =
//   lgmath::se3::vec2jacinv(xi_21);

//   const double T = (knot2->time() - knot1->time()).seconds();
//   const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6,
//   1>::Ones(); const Matrix18d Qinv_T = interface_->getQinvPublic(T, ones);
//   const Matrix18d Tran_T = interface_->getTranPublic(T);

//   lgmath::se3::Transformation T_i0_prev = T1;
//   Eigen::Matrix<double, 6, 1> w_i_prev = w1;
//   Eigen::Matrix<double, 6, 1> dw_i_prev = dw1;
//   auto it = time_to_index_map.begin();
//   it++;
//   for (; it != time_to_index_map.end(); ++it) {
//     const Time &time = it->first;
//     const uint state_index = it->second;

//     // Get Lambda, Omega for this time
//     const double tau = (time - time1_).seconds();
//     const double kappa = (time2_ - time).seconds();
//     const Matrix18d Q_tau = interface_->getQPublic(tau, ones);
//     const Matrix18d Tran_kappa = interface_->getTranPublic(kappa);
//     const Matrix18d Tran_tau = interface_->getTranPublic(tau);
//     const Matrix18d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T);
//     const Matrix18d lambda = (Tran_tau - omega * Tran_T);

//     // Calculate interpolated relative se3 algebra
//     const Eigen::Matrix<double, 6, 1> xi_i1 =
//         lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
//         omega.block<6, 6>(0, 0) * xi_21 +
//         omega.block<6, 6>(0, 6) * J_21_inv * w2 +
//         omega.block<6, 6>(0, 12) *
//             (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv *
//             dw2);
//     const Eigen::Matrix<double, 6, 1> xi_j1 =
//         lambda.block<6, 6>(6, 6) * w1 + lambda.block<6, 6>(6, 12) * dw1 +
//         omega.block<6, 6>(6, 0) * xi_21 +
//         omega.block<6, 6>(6, 6) * J_21_inv * w2 +
//         omega.block<6, 6>(6, 12) *
//             (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv *
//             dw2);
//     const Eigen::Matrix<double, 6, 1> xi_k1 =
//         lambda.block<6, 6>(12, 6) * w1 + lambda.block<6, 6>(12, 12) * dw1 +
//         omega.block<6, 6>(12, 0) * xi_21 +
//         omega.block<6, 6>(12, 6) * J_21_inv * w2 +
//         omega.block<6, 6>(12, 12) *
//             (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv *
//             dw2);

//     const lgmath::se3::Transformation T_i1(xi_i1);
//     const lgmath::se3::Transformation T_i0 = T_i1 * T1;  // interpolated pose
//     const auto w_i =
//         lgmath::se3::vec2jac(xi_i1) * xi_j1;  // interpolated velocity
//     const auto dw_i = lgmath::se3::vec2jac(xi_i1) *
//                       (xi_k1 + 0.5 * lgmath::se3::curlyhat(xi_j1) *
//                                    w_i);  // interpolated acceleration

//     // Get motion prior Jacobians
//     Matrix18d Phi = interface_->getTranPublic((time - tprev).seconds());
//     Matrix18d F = Matrix18d::Zero();
//     Matrix18d E = Matrix18d::Zero();
//     getMotionPriorJacobians_(T_i0_prev, T_i0, w_i, dw_i, Phi, F, E);

//     //
//     tprev = time;
//     T_i0_prev = T_i0;
//     w_i_prev = w_i;
//     dw_i_prev = dw_i;
//   }

//   // Accelerometer Error Terms:
//   using ErrorType = Eigen::Matrix<double, 3, 1>;
//   using JacType = Eigen::Matrix<double, 3, 6>;
//   for (AccelerationErrorEvaluator::ConstPtr &accel_err : accel_err_vec_) {
//     const ErrorType raw_error = accel_err->value();
//     const ErrorType white_error = acc_noise_model_->whitenError(raw_error);
//     const double sqrt_w = sqrt(acc_loss_func_->weight(white_error.norm()));
//     const ErrorType error = sqrt_w * white_error;

//     const uint state_index = time_to_index_map.at(accel_err->getTime());
//     JacType G_pose, G_accel, G_bias, G_T_mi;
//     G_pose.setZero();
//     G_accel.setZero();
//     G_bias.setZero();
//     G_T_mi.setZero();
//     accel_err->getMeasJacobians(G_pos, G_accel, G_bias, G_T_mi);
//     // weight and whiten Jacobians
//     G_pos = sqrt_w * acc_noise_model_->getSqrtInformation * G_pos;
//     G_accel = sqrt_w * acc_noise_model_->getSqrtInformation * G_accel;
//     G_bias = sqrt_w * acc_noise_model_->getSqrtInformation * G_bias;
//     G_T_mi = sqrt_w * acc_noise_model_->getSqrtInformation * G_T_mi;

//     std::map<int, JacType> jacobians;

//     jacobians[state_index + pose_blk_offset] = G_pose;
//     jacobians[state_index + acceleration_blk_offset] = G_accel;

//     int index = state_index;
//     if (options_.bias_const_over_window) {
//       index = init_state_index + bias_blk_offset;
//     } else {
//       index = state_index + bias_blk_offset;
//     }
//     jacobians[index] = G_bias;

//     if (options_.estimate_T_mi) {
//       if (options_.T_mi_const_over_window) {
//         index = init_state_index + T_mi_blk_offset;
//       } else {
//         index = state_index + T_mi_blk_offset;
//       }
//       jacobians[index] = G_T_mi;
//     }

//     for (auto it = jacobians.begin(); it != jacobians.end(); ++it) {
//       const uint blkIdx1 = it->first;
//       const auto &jac1 = it->second;
//       const Eigen::MatrixXd newGradTerm = (-1) * jac1.transpose() * error;
//       { b->mapAt(blkIdx1) += newGradTerm; }

//       for (auto it2 = it; it2 != jacobians.end(); ++it2) {
//         const uint blkIdx2 = it2->first;
//         const auto &jac2 = it2->second;
//         uint row, col;
//         const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
//           if (blkIdx1 <= blkIdx2) {
//             row = blkIdx1;
//             col = blkIdx2;
//             return jac1.transpose() * jac2;
//           } else {
//             row = blkIdx2;
//             col = blkIdx1;
//             return jac2.transpose() * jac1;
//           }
//         }();
//         BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col,
//         true); entry.data += newHessianTerm;
//       }
//     }
//   }

//   for (GyroErrorEvaluator::ConstPtr &gyro_err : gyro_err_vec_) {
//     const ErrorType raw_error = gyro_err->value();
//     const ErrorType white_error = gyro_noise_model_->whitenError(raw_error);
//     const double sqrt_w = sqrt(gyro_loss_func_->weight(white_error.norm()));
//     const ErrorType error = sqrt_w * white_error;

//     const uint state_index = time_to_index_map.at(gyro_err->getTime());
//     JacType G_vel, G_bias;
//     G_vel.setZero();
//     G_bias.setZero();
//     gyro_err->getMeasJacobians(G_vel, G_bias);
//     G_vel = sqrt_w * gyro_noise_model_->getSqrtInformation * G_vel;
//     G_vias = sqrt_w * gyro_noise_model_->getSqrtInformation * G_bias;

//     std::map<int, JacType> jacobians;

//     jacobians[state_index + velocity_blk_offset] = G_vel;

//     int index = state_index;
//     if (options_.bias_const_over_window) {
//       index = init_state_index + bias_blk_offset;
//     } else {
//       index = state_index + bias_blk_offset;
//     }
//     jacobians[index] = G_bias;

//     for (auto it = jacobians.begin(); it != jacobians.end(); ++it) {
//       const uint blkIdx1 = it->first;
//       const auto &jac1 = it->second;
//       const Eigen::MatrixXd newGradTerm = (-1) * jac1.transpose() * error;
//       { b->mapAt(blkIdx1) += newGradTerm; }

//       for (auto it2 = it; it2 != jacobians.end(); ++it2) {
//         const uint blkIdx2 = it2->first;
//         const auto &jac2 = it2->second;
//         uint row, col;
//         const Eigen::MatrixXd newHessianTerm = [&]() -> Eigen::MatrixXd {
//           if (blkIdx1 <= blkIdx2) {
//             row = blkIdx1;
//             col = blkIdx2;
//             return jac1.transpose() * jac2;
//           } else {
//             row = blkIdx2;
//             col = blkIdx1;
//             return jac2.transpose() * jac1;
//           }
//         }();
//         BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(row, col,
//         true); entry.data += newHessianTerm;
//       }
//     }
//   }

//   // todo: create map of previously computed Jacobians (speedup)
//   std::map<Time, JacType> p2p_jac_map;
//   for (int i = 0; i < p2p_err_vec.size(); ++i) {
//     const P2PErrorEvaluator::ConstPtr &p2p_err = p2p_err_vec_[i];
//     const StaticNoiseModel::ConstPtr noise_model = p2p_noise_models_[i];

//     const ErrorType raw_error = p2p_err->value();
//     const ErrorType white_error = noise_model->whitenError(raw_error);
//     const double sqrt_w = sqrt(p2p_loss_func_->weight(white_error.norm()));
//     const ErrorType error = sqrt_w * white_error;

//     const uint index = time_to_index_map.at(p2p_err->getTime());
//     JacType G;
//     if (auto search = p2p_jac_map.find(p2p_err->getTime());
//         search != p2p_jac_map.end()) {
//       G = search->second;
//     } else {
//       G = p2p_err->getJacobianPose();
//       p2p_jac_map[p2p_err->getTime()] = G;
//     }

//     const JacType G2 = sqrt_w * noise_model->getSqrtInformation() * G;

//     const Eigen::MatrixXd newGradTerm = (-1) * G2.transpose() * error;
//     { b->mapAt(index) += newGradTerm; }
//     const Eigen::MatrixXd newHessianTerm = G2.transpose() * G2;
//     BlockSparseMatrix::BlockRowEntry &entry = A->rowEntryAt(index, index,
//     true); entry.data += newHessianTerm;
//   }
// };

// }  // namespace steam
