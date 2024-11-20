#include "steam/problem/cost_term/preintegrated_imu_cost_term.hpp"
#include <iostream>

namespace steam {

// indices of the different perturbations in state vector X
#define IP1 0
#define IR1 3
#define IV1 6
#define IP2 9
#define IR2 12
#define IV2 15
#define IBA 18
#define IBG 21

PreintIMUCostTerm::Ptr PreintIMUCostTerm::MakeShared(
    const Time time1,
    const Time time2,
    const Evaluable<PoseType>::ConstPtr &transform_r_to_m_1,
    const Evaluable<PoseType>::ConstPtr &transform_r_to_m_2,
    const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m_1,
    const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m_2,
    const Evaluable<BiasType>::ConstPtr &bias,
    const Options &options) {
  return std::make_shared<PreintIMUCostTerm>(
    time1,
    time2,
    transform_r_to_m_1,
    transform_r_to_m_2,
    v_m_to_r_in_m_1,
    v_m_to_r_in_m_2,
    bias,
    options);
}

/** \brief Compute the cost to the objective function */
double PreintIMUCostTerm::cost() const {
  const PreintegratedMeasurement preint_meas = preintegrate_();
  const Eigen::Matrix<double, 9, 1> raw_error = get_error();
  StaticNoiseModel<9>::Ptr noise_model = StaticNoiseModel<9>::MakeShared(preint_meas.cov);
  const auto cost = loss_func_->cost(noise_model->getWhitenedErrorNorm(raw_error));
  return cost;
}

/** \brief Get keys of variables related to this cost term */
void PreintIMUCostTerm::getRelatedVarKeys(KeySet &keys) const {
  transform_r_to_m_1_->getRelatedVarKeys(keys);
  v_m_to_r_in_m_1_->getRelatedVarKeys(keys);
  transform_r_to_m_1_->getRelatedVarKeys(keys);
  v_m_to_r_in_m_2_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
}

PreintegratedMeasurement PreintIMUCostTerm::preintegrate_() const {
  Eigen::Matrix3d C_ij = Eigen::Matrix3d::Identity();
  Eigen::Vector3d r_ij = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_ij = Eigen::Vector3d::Zero();

  const Eigen::Matrix<double, 6, 1> b = bias_->forward()->value();
  const Eigen::Vector3d ba = b.block<3, 1>(0, 0);
  const Eigen::Vector3d bg = b.block<3, 1>(3, 0);
  // cov: delta_phi, delta_p, delta_v
  Eigen::Matrix<double, 9, 9> cov = Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
  Q.block<3, 3>(0, 0) = R_acc_;
  Q.block<3, 3>(3, 3) = R_ang_;

  for (size_t i = 0; i < imu_data_vec_.size(); ++i) {
    const IMUData &imu_data = imu_data_vec_[i];
    double delta_t = 0;
    if (i == 0) {
      delta_t = imu_data.timestamp - time1_.seconds();
      const Eigen::Vector3d phi_i = (imu_data.ang_vel - bg) * delta_t;
      const Eigen::Matrix3d C_i_i_plus_1 = lgmath::so3::vec2rot(phi_i);
      r_ij += v_ij * delta_t + 0.5 * C_ij * (imu_data.lin_acc - ba) * delta_t * delta_t;
      v_ij += C_ij * (imu_data.lin_acc - ba) * delta_t;
      C_ij = C_ij * C_i_i_plus_1;
      Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
      // note: we multiply phi_i by (-1) to convert left Jacobian to right Jacobian
      B.block<3, 3>(0, 3) = lgmath::so3::vec2jac(-phi_i) * delta_t;
      B.block<3, 3>(3, 0) = 0.5 * C_ij * delta_t * delta_t;
      B.block<3, 3>(6, 0) = C_ij * delta_t;
      cov = B * Q * B.transpose();
    }
    if (i < imu_data_vec_.size() - 1) {
      delta_t = imu_data_vec_[i + 1].timestamp - imu_data_vec_[i].timestamp;
    } else if (i == imu_data_vec_.size() - 1) {
      delta_t = time2_.seconds() - imu_data_vec_[i].timestamp;
    }
    const Eigen::Vector3d phi_i = (imu_data.ang_vel - bg) * delta_t;
    const Eigen::Matrix3d C_i_i_plus_1 = lgmath::so3::vec2rot(phi_i);
    r_ij += v_ij * delta_t + 0.5 * C_ij * (imu_data.lin_acc - ba) * delta_t * delta_t;
    v_ij += C_ij * (imu_data.lin_acc - ba) * delta_t;
    C_ij = C_ij * C_i_i_plus_1;
    Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
    // note: we multiply phi_i by (-1) to convert left Jacobian to right Jacobian
    B.block<3, 3>(0, 3) = lgmath::so3::vec2jac(-phi_i) * delta_t;
    B.block<3, 3>(3, 0) = 0.5 * C_ij * delta_t * delta_t;
    B.block<3, 3>(6, 0) = C_ij * delta_t;
    Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
    A.block<3, 3>(0, 0) = C_i_i_plus_1.transpose();
    A.block<3, 3>(3, 0) = -0.5 * C_ij * lgmath::so3::hat(imu_data.lin_acc - ba) * delta_t * delta_t;
    A.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * delta_t;
    A.block<3, 3>(6, 0) = -C_ij * lgmath::so3::hat(imu_data.lin_acc - ba) * delta_t;
    cov = A * cov * A.transpose() + B * Q * B.transpose();
  }
  return PreintegratedMeasurement(C_ij, r_ij, v_ij, cov);
}

Eigen::Matrix<double, 9, 1> PreintIMUCostTerm::get_error() const {
  const PreintegratedMeasurement preint_meas = preintegrate_();
  const Eigen::Matrix3d C_ij = preint_meas.C_ij;
  const Eigen::Vector3d r_ij = preint_meas.r_ij;
  const Eigen::Vector3d v_ij = preint_meas.v_ij;

  const double delta_t_i_j = (time2_ - time1_).seconds();
  const auto T_i = transform_r_to_m_1_->forward()->value();
  const auto T_j = transform_r_to_m_2_->forward()->value();
  const Eigen::Matrix3d C_i = T_i.C_ba();
  const Eigen::Vector3d p_i = T_i.r_ab_inb();
  const Eigen::Vector3d v_i = v_m_to_r_in_m_1_->forward()->value();
  const Eigen::Matrix3d C_j = T_j.C_ba();
  const Eigen::Vector3d p_j = T_j.r_ab_inb();
  const Eigen::Vector3d v_j = v_m_to_r_in_m_2_->forward()->value();

  Eigen::Matrix<double, 9, 1> error = Eigen::Matrix<double, 9, 1>::Zero();
  // std::cout << "C_ij: " << C_ij << std::endl;
  // std::cout << "C_i.transpose() * C_j " << C_i.transpose() * C_j << std::endl;
  // std::cout << "C_i.transpose() * (p_j - p_i - v_i * delta_t_i_j - 0.5 * gravity_ * pow(delta_t_i_j, 2)) " << C_i.transpose() * (p_j - p_i - v_i * delta_t_i_j - 0.5 * gravity_ * pow(delta_t_i_j, 2)) << std::endl;
  // std::cout << "r_ij " << r_ij << std::endl;
  // std::cout << "C_i.transpose() * (v_j - v_i - gravity_ * delta_t_i_j) " << C_i.transpose() * (v_j - v_i - gravity_ * delta_t_i_j) << std::endl;
  // std::cout << "v_ij " << v_ij << std::endl;
  error.block<3, 1>(0, 0) = lgmath::so3::rot2vec(C_ij.transpose() * C_i.transpose() * C_j);
  error.block<3, 1>(3, 0) = C_i.transpose() * (p_j - p_i - v_i * delta_t_i_j - 0.5 * gravity_ * pow(delta_t_i_j, 2)) - r_ij;
  error.block<3, 1>(6, 0) = C_i.transpose() * (v_j - v_i - gravity_ * delta_t_i_j) - v_ij;
  return error;
}

Eigen::Matrix<double, 9, 24> PreintIMUCostTerm::get_jacobian() const {
  Eigen::Matrix3d C_ij = Eigen::Matrix3d::Identity();
  const Eigen::Matrix<double, 6, 1> b = bias_->forward()->value();
  const Eigen::Vector3d ba = b.block<3, 1>(0, 0);
  const Eigen::Vector3d bg = b.block<3, 1>(3, 0);

  Eigen::Matrix3d drij_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dvij_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dvij_dbg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dpij_dba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dpij_dbg = Eigen::Matrix3d::Zero();

  for (size_t i = 0; i < imu_data_vec_.size(); ++i) {
      const IMUData &imu_data = imu_data_vec_[i];
      double delta_t = 0;
      if (i == 0) {
        delta_t = imu_data.timestamp - time1_.seconds();
        const Eigen::Vector3d phi_i = (imu_data.ang_vel - bg) * delta_t; 
        dpij_dba += dvij_dba * delta_t - 0.5 * C_ij * delta_t * delta_t;
        dpij_dbg += dvij_dbg * delta_t - 0.5 * C_ij * lgmath::so3::hat(imu_data.lin_acc - ba) * drij_dbg * delta_t * delta_t;
        dvij_dba -= C_ij * delta_t;
        dvij_dbg -= C_ij * lgmath::so3::hat(imu_data.lin_acc - ba) * drij_dbg * delta_t;
        C_ij = C_ij * lgmath::so3::vec2rot(phi_i);
        // note: we multiply phi_i by (-1) to convert left Jacobian to right Jacobian
        drij_dbg -= lgmath::so3::vec2jac(-phi_i) * delta_t;
      }
      if (i < imu_data_vec_.size() - 1) {
        delta_t = imu_data_vec_[i + 1].timestamp - imu_data_vec_[i].timestamp;
      } else if (i == imu_data_vec_.size() - 1) {
        delta_t = time2_.seconds() - imu_data_vec_[i].timestamp;
      }
      const Eigen::Vector3d phi_i = (imu_data.ang_vel - bg) * delta_t;
      const Eigen::Matrix3d C_i_i_plus_1 = lgmath::so3::vec2rot(phi_i);
      dpij_dba += dvij_dba * delta_t - 0.5 * C_ij * delta_t * delta_t;
      dpij_dbg += dvij_dbg * delta_t - 0.5 * C_ij * lgmath::so3::hat(imu_data.lin_acc - ba) * drij_dbg * delta_t * delta_t;
      dvij_dba -= C_ij * delta_t;
      dvij_dbg -= C_ij * lgmath::so3::hat(imu_data.lin_acc - ba) * drij_dbg * delta_t;
      C_ij = C_ij * C_i_i_plus_1;
      // note: we multiply phi_i by (-1) to convert left Jacobian to right Jacobian
      drij_dbg = C_i_i_plus_1.transpose() * drij_dbg - lgmath::so3::vec2jac(-phi_i) * delta_t;
  }

  Eigen::Matrix<double, 9, 24> J = Eigen::Matrix<double, 9, 24>::Zero();
  const double delta_t_i_j = (time2_ - time1_).seconds();
  const lgmath::se3::Transformation T_i = transform_r_to_m_1_->forward()->value();
  const lgmath::se3::Transformation T_j = transform_r_to_m_2_->forward()->value();
  const Eigen::Matrix3d C_i = T_i.C_ba();
  const Eigen::Vector3d p_i = T_i.r_ab_inb();
  const Eigen::Vector3d v_i = v_m_to_r_in_m_1_->forward()->value();
  const Eigen::Matrix3d C_j = T_j.C_ba();
  const Eigen::Vector3d p_j = T_j.r_ab_inb();
  const Eigen::Vector3d v_j = v_m_to_r_in_m_2_->forward()->value();

  const Eigen::Vector3d e_rij = lgmath::so3::rot2vec(C_ij.transpose() * C_i.transpose() * C_j);
  // e(r)
  // note: we multiply e_rij by (-1) to convert left Jacobian inverse to right Jacobian inverse
  J.block<3, 3>(0, IR1) = -lgmath::so3::vec2jacinv(-e_rij) * C_j.transpose() * C_i;
  J.block<3, 3>(0, IR2) = lgmath::so3::vec2jacinv(-e_rij);
  // we use the left Jacobian on purpose here
  J.block<3, 3>(0, IBG) = -lgmath::so3::vec2jacinv(e_rij) * drij_dbg;
  // e(p)
  J.block<3, 3>(3, IR1) = lgmath::so3::hat(C_i.transpose() * (p_j - p_i - v_i * delta_t_i_j - 0.5 * gravity_ * delta_t_i_j * delta_t_i_j));
  J.block<3, 3>(3, IP1) = -Eigen::Matrix3d::Identity();
  // J.block<3, 3>(3, IV1) = -C_i.transpose() * delta_t_i_j;
  J.block<3, 3>(3, IV1) = -Eigen::Matrix3d::Identity() * delta_t_i_j;
  J.block<3, 3>(3, IP2) = C_i.transpose() * C_j;
  J.block<3, 3>(3, IBA) = -dpij_dba;
  J.block<3, 3>(3, IBG) = -dpij_dbg;
  // e(v)
  J.block<3, 3>(6, IR1) = lgmath::so3::hat(C_i.transpose() * (v_j - v_i - gravity_ * delta_t_i_j));
  // J.block<3, 3>(6, IV1) = -C_i.transpose();
  J.block<3, 3>(6, IV1) = -Eigen::Matrix3d::Identity();
  // J.block<3, 3>(6, IV2) = C_i.transpose();
  J.block<3, 3>(6, IV2) = C_i.transpose() * C_j;
  J.block<3, 3>(6, IBA) = -dvij_dba;
  J.block<3, 3>(6, IBG) = -dvij_dbg;

  return J;
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void PreintIMUCostTerm::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  using namespace steam::se3;
  using namespace steam::vspace;

  const auto T1_ = transform_r_to_m_1_->forward();
  const auto v1_ = v_m_to_r_in_m_1_->forward();
  const auto T2_ = transform_r_to_m_2_->forward();
  const auto v2_ = v_m_to_r_in_m_2_->forward();
  const auto b_ = bias_->forward();

  Eigen::Matrix<double, 24, 24> A = Eigen::Matrix<double, 24, 24>::Zero();
  Eigen::Matrix<double, 24, 1> c = Eigen::Matrix<double, 24, 1>::Zero();

  Eigen::Matrix<double, 9, 24> G = get_jacobian();
  const PreintegratedMeasurement preint_meas = preintegrate_();
  const Eigen::Matrix<double, 9, 1> raw_error = get_error();
  StaticNoiseModel<9>::Ptr noise_model = StaticNoiseModel<9>::MakeShared(preint_meas.cov);

  const Eigen::Matrix<double, 9, 1> white_error = noise_model->whitenError(raw_error);
  const double sqrt_w = sqrt(loss_func_->weight(white_error.norm()));
  const Eigen::Matrix<double, 9, 1> error = sqrt_w * white_error;

  G = sqrt_w * noise_model->getSqrtInformation() * G;
  A = G.transpose() * G;
  c = (-1) * G.transpose() * error;
  
  std::vector<bool> active;
  active.push_back(transform_r_to_m_1_->active());
  active.push_back(v_m_to_r_in_m_1_->active());
  active.push_back(transform_r_to_m_2_->active());
  active.push_back(v_m_to_r_in_m_2_->active());
  active.push_back(bias_->active());

  std::vector<StateKey> keys;

  if (active[0]) {
    const auto T1node = std::static_pointer_cast<Node<PoseType>>(T1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    transform_r_to_m_1_->backward(lhs, T1node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[1]) {
    const auto v1node = std::static_pointer_cast<Node<VelType>>(v1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    v_m_to_r_in_m_1_->backward(lhs, v1node, jacs);
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
    transform_r_to_m_2_->backward(lhs, T2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[3]) {
    const auto v2node = std::static_pointer_cast<Node<VelType>>(v2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    v_m_to_r_in_m_2_->backward(lhs, v2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[4]) {
    const auto bnode = std::static_pointer_cast<Node<BiasType>>(b_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    bias_->backward(lhs, bnode, jacs);
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

  std::vector<int> blk_indices = {0, 6, 9, 15, 18};
  std::vector<int> blk_sizes = {6, 3, 6, 3, 6};


  for (int i = 0; i < 5; ++i) {
    if (!active[i]) continue;
    // Get the key and state range affected
    const auto &key1 = keys[i];
    unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

    // Calculate terms needed to update the right-hand-side
    const Eigen::MatrixXd newGradTerm = [&]() -> Eigen::MatrixXd {
      if (blk_sizes[i] == 3) {
        return c.block<3, 1>(blk_indices[i], 0);
      } else {
        return c.block<6, 1>(blk_indices[i], 0);
      }
    }();
    // Update the right-hand side (thread critical)
#pragma omp critical(b_update)
    { gradient_vector->mapAt(blkIdx1) += newGradTerm; }

    for (int j = i; j < 5; ++j) {
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
          if (blk_sizes[i] == 3 && blk_sizes[j] == 3) {
            return A.block<3, 3>(blk_indices[i], blk_indices[j]);
          } else if (blk_sizes[i] == 3 && blk_sizes[j] == 6) {
            return A.block<3, 6>(blk_indices[i], blk_indices[j]);
          } else if (blk_sizes[i] == 6 && blk_sizes[j] == 3) {
            return A.block<6, 3>(blk_indices[i], blk_indices[j]);
          } else if (blk_sizes[i] == 6 && blk_sizes[j] == 6) {
            return A.block<6, 6>(blk_indices[i], blk_indices[j]);
          } else {
            throw std::runtime_error("invalid block size");
          }
          
        } else {
          row = blkIdx2;
          col = blkIdx1;
          if (blk_sizes[i] == 3 && blk_sizes[j] == 3) {
            return A.block<3, 3>(blk_indices[j], blk_indices[i]);
          } else if (blk_sizes[j] == 3 && blk_sizes[i] == 6) {
            return A.block<3, 6>(blk_indices[j], blk_indices[i]);
          } else if (blk_sizes[j] == 6 && blk_sizes[i] == 3) {
            return A.block<6, 3>(blk_indices[j], blk_indices[i]);
          } else if (blk_sizes[j] == 6 && blk_sizes[i] == 6) {
            return A.block<6, 6>(blk_indices[j], blk_indices[i]);
          } else {
            throw std::runtime_error("invalid block size");
          }
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
