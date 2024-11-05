#include <iostream>

#include "steam/problem/cost_term/p2p_global_perturb_super_cost_term.hpp"

namespace steam {

P2PGlobalSuperCostTerm::Ptr P2PGlobalSuperCostTerm::MakeShared(
    const Time time,
    const Evaluable<PoseType>::ConstPtr &transform_r_to_m,
    const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m,
    const Evaluable<BiasType>::ConstPtr &bias,
    const Options &options,
    const std::vector<steam::IMUData> &imu_data_vec) {
  return std::make_shared<P2PGlobalSuperCostTerm>(time, transform_r_to_m, v_m_to_r_in_m, bias, options, imu_data_vec);
}

/** \brief Compute the cost to the objective function */
double P2PGlobalSuperCostTerm::cost() const {
  double cost = 0;

  const double rinv = 1.0 / options_.r_p2p;
  const double sqrt_rinv = sqrt(rinv);

  const std::vector<IntegratedState> states = integrate_(false);
  std::vector<double> pose_times;
  for (const IntegratedState &state : states) {
    pose_times.push_back(state.timestamp);
  }

#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : cost)
  for (unsigned int i = 0; i < meas_times_.size(); ++i) {
    const double &ts = meas_times_[i];
    const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);
    // interpolation
    int start_index = 0, end_index = 0;
    for (size_t k = 0; k < pose_times.size(); ++k) {
      if (pose_times[k] > ts) {
        end_index = k;
        break;
      }
      start_index = k;
      end_index = k;
    }
    Eigen::Matrix3d C_mr = states[0].C_mr;
    Eigen::Vector3d r_rm_in_m = states[0].r_rm_in_m;
    double alpha = 0;
    if ((ts == pose_times[start_index]) || (start_index == end_index) || (pose_times[start_index] == pose_times[end_index])) {
      alpha = 0;
    } else if (ts == pose_times[end_index]) {
      alpha = 1;
    } else {
      alpha = (ts - pose_times[start_index]) / (pose_times[end_index] - pose_times[start_index]);
    }
    r_rm_in_m = alpha * states[end_index].r_rm_in_m + (1 - alpha) * states[start_index].r_rm_in_m;
    const Eigen::Vector3d phi = lgmath::so3::rot2vec(states[start_index].C_mr.transpose() * states[end_index].C_mr);
    C_mr = states[start_index].C_mr * lgmath::so3::vec2rot(alpha * phi);

    for (const int &match_idx : bin_indices) {
      const auto &p2p_match = p2p_matches_.at(match_idx);
      const double raw_error =
          p2p_match.normal.transpose() *
          (p2p_match.reference - C_mr * p2p_match.query - r_rm_in_m);
      cost += p2p_loss_func_->cost(sqrt_rinv * fabs(raw_error));
    }
  }
  return cost;
}

/** \brief Get keys of variables related to this cost term */
void P2PGlobalSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
  transform_r_to_m_->getRelatedVarKeys(keys);
  v_m_to_r_in_m_->getRelatedVarKeys(keys);
  bias_->getRelatedVarKeys(keys);
}

void P2PGlobalSuperCostTerm::initP2PMatches() {
  p2p_match_bins_.clear();
  for (int i = 0; i < (int)p2p_matches_.size(); ++i) {
    const auto &p2p_match = p2p_matches_.at(i);
    const auto &timestamp = p2p_match.timestamp;
    if (p2p_match_bins_.find(timestamp) == p2p_match_bins_.end()) {
      p2p_match_bins_[timestamp] = {i};
    } else {
      p2p_match_bins_[timestamp].push_back(i);
    }
  }
  meas_times_.clear();
  for (auto it = p2p_match_bins_.begin(); it != p2p_match_bins_.end(); it++) {
    meas_times_.push_back(it->first);
  }
  min_point_time_ = *std::min_element(meas_times_.begin(), meas_times_.end());
  max_point_time_ = *std::max_element(meas_times_.begin(), meas_times_.end());
  // initialize_interp_matrices_();
}

std::vector<IntegratedState> P2PGlobalSuperCostTerm::integrate_(bool compute_jacobians) const {
  const auto T_mr = transform_r_to_m_->forward()->value();
  const Eigen::Matrix3d C_i = T_mr.C_ba();
  const Eigen::Vector3d p_i = T_mr.r_ab_inb();
  const Eigen::Vector3d v_i = v_m_to_r_in_m_->forward()->value();
  const Eigen::Matrix<double, 6, 1> b = bias_->forward()->value();
  const Eigen::Vector3d ba = b.block<3, 1>(0, 0);
  const Eigen::Vector3d bg = b.block<3, 1>(3, 0);
  
  std::vector<IntegratedState> integrated_states;

  // initial state
  IntegratedState state = IntegratedState(C_i, p_i, v_i, curr_time_);
  if (compute_jacobians) {
    // [dp / d delta x; dC / d delta x], x = (pos, rot, vel, b_a, b_g)
    Eigen::Matrix<double, 6, 15> jacobian = Eigen::Matrix<double, 6, 15>::Zero();
    jacobian.block<3, 3>(0, 0) = C_i;
    jacobian.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    state.jacobian = jacobian;
  }
  integrated_states.push_back(state);

  // before: (integrate backwards in time)
  if (imu_before.size() > 0) {
    Eigen::Matrix3d C_mr = C_i;
    Eigen::Vector3d r_rm_in_m = p_i;
    Eigen::Vector3d v_rm_in_m = v_i;
    Eigen::Matrix3d C_ik = Eigen::Matrix3d::Identity();
    // Jacobians
    Eigen::Matrix3d drk_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dvk_dri = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dvk_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dvk_dba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dpk_dri = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dpk_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dpk_dba = Eigen::Matrix3d::Zero();
    double delta_t_ik = 0;
    double delta_t = fabs(imu_before[0].timestamp - curr_time_);
    if (delta_t > 1.0e-6) {
      uint k = 0;
      delta_t_ik += delta_t;
      const Eigen::Vector3d phi_k = -1 * (imu_before[k].ang_vel - bg) * delta_t;
      const Eigen::Matrix3d C_k_k_plus_1 = lgmath::so3::vec2rot(phi_k);

      C_mr = C_mr * C_k_k_plus_1;
      v_rm_in_m -= ((C_mr * (imu_before[k].lin_acc - ba) + gravity_) * delta_t);
      r_rm_in_m -= (v_rm_in_m * delta_t + 0.5 * (C_mr * (imu_before[k].lin_acc - ba) + gravity_) * delta_t * delta_t);
      const Eigen::Matrix3d C_k = C_mr;

      IntegratedState state = IntegratedState(C_mr, r_rm_in_m, v_rm_in_m, imu_before[k].timestamp);
      if (compute_jacobians) {
        // note: we multiply phi_k by (-1) to convert left Jacobian to right Jacobian
        drk_dbg = C_k_k_plus_1.transpose() * drk_dbg + lgmath::so3::vec2jac(-phi_k) * delta_t;
        C_ik = C_ik * C_k_k_plus_1;
        dvk_dri += C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * C_ik.transpose() * delta_t;
        dvk_dbg += C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * drk_dbg * delta_t;
        dvk_dba += C_k * delta_t;
        dpk_dri += -dvk_dri * delta_t + 0.5 * C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * C_ik.transpose() * delta_t * delta_t;
        dpk_dbg += -dvk_dbg * delta_t + 0.5 * C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * drk_dbg * delta_t * delta_t;
        dpk_dba += -dvk_dba * delta_t + 0.5 * C_k * delta_t * delta_t;
        // [dp / d delta x; dC / d delta x], x = (pos, rot, vel, b_a, b_g)
        Eigen::Matrix<double, 6, 15> jacobian = Eigen::Matrix<double, 6, 15>::Zero();
        // dp / d delta x 
        jacobian.block<3, 3>(0, 0) = C_i;
        jacobian.block<3, 3>(0, 3) = dpk_dri;
        jacobian.block<3, 3>(0, 6) = -C_i * delta_t_ik;
        jacobian.block<3, 3>(0, 9) = dpk_dba;
        jacobian.block<3, 3>(0, 12) = dpk_dbg;
        // dC / d delta x
        jacobian.block<3, 3>(3, 3) = C_ik.transpose();
        jacobian.block<3, 3>(3, 12) = drk_dbg;

        state.jacobian = jacobian;
      }
      integrated_states.push_back(state);
    }
    for (size_t k = 0; k < imu_before.size(); ++k) {
      double delta_t = 0;
      double pose_time = 0;
      if (k < imu_before.size() - 1) {
        delta_t = fabs(imu_before[k + 1].timestamp - imu_before[k].timestamp);
        pose_time = imu_before[k + 1].timestamp;
      } else if (k == imu_before.size() - 1) {
        delta_t = fabs(min_point_time_ - imu_before[k].timestamp);
        pose_time = min_point_time_;
      }
      assert(delta_t > 0);

      delta_t_ik += delta_t;
      const Eigen::Vector3d phi_k = -1 * (imu_before[k].ang_vel - bg) * delta_t;
      const Eigen::Matrix3d C_k_k_plus_1 = lgmath::so3::vec2rot(phi_k);

      C_mr = C_mr * C_k_k_plus_1;
      v_rm_in_m -= (C_mr * (imu_before[k].lin_acc - ba) + gravity_) * delta_t;
      r_rm_in_m -= v_rm_in_m * delta_t + 0.5 * (C_mr * (imu_before[k].lin_acc - ba) + gravity_) * delta_t * delta_t;
      const Eigen::Matrix3d C_k = C_mr;

      IntegratedState state = IntegratedState(C_mr, r_rm_in_m, v_rm_in_m, pose_time);
      if (compute_jacobians) {
        // note: we multiply phi_k by (-1) to convert left Jacobian to right Jacobian
        drk_dbg = C_k_k_plus_1.transpose() * drk_dbg + lgmath::so3::vec2jac(-phi_k) * delta_t;
        C_ik = C_ik * C_k_k_plus_1;
        dvk_dri += C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * C_ik.transpose() * delta_t;
        dvk_dbg += C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * drk_dbg * delta_t;
        dvk_dba += C_k * delta_t;
        dpk_dri += -dvk_dri * delta_t + 0.5 * C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * C_ik.transpose() * delta_t * delta_t;
        dpk_dbg += -dvk_dbg * delta_t + 0.5 * C_k * lgmath::so3::hat(imu_before[k].lin_acc - ba) * drk_dbg * delta_t * delta_t;
        dpk_dba += -dvk_dba * delta_t + 0.5 * C_k * delta_t * delta_t;
        // [dp / d delta x; dC / d delta x], x = (pos, rot, vel, b_a, b_g)
        Eigen::Matrix<double, 6, 15> jacobian = Eigen::Matrix<double, 6, 15>::Zero();
        // dp / d delta x 
        jacobian.block<3, 3>(0, 0) = C_i;
        jacobian.block<3, 3>(0, 3) = dpk_dri;
        jacobian.block<3, 3>(0, 6) = -C_i * delta_t_ik;
        jacobian.block<3, 3>(0, 9) = dpk_dba;
        jacobian.block<3, 3>(0, 12) = dpk_dbg;
        // dC / d delta x
        jacobian.block<3, 3>(3, 3) = C_ik.transpose();
        jacobian.block<3, 3>(3, 12) = drk_dbg;

        state.jacobian = jacobian;
      }
      integrated_states.push_back(state);
    }
  }

  // after:
  if (imu_after.size() > 0) {
    Eigen::Matrix3d C_mr = C_i;
    Eigen::Vector3d r_rm_in_m = p_i;
    Eigen::Vector3d v_rm_in_m = v_i;
    Eigen::Matrix3d C_ik = Eigen::Matrix3d::Identity();
    // Jacobians
    Eigen::Matrix3d drk_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dvk_dri = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dvk_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dvk_dba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dpk_dri = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dpk_dbg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d dpk_dba = Eigen::Matrix3d::Zero();
    double delta_t_ik = 0;
    double delta_t = imu_after[0].timestamp - curr_time_;
    if (delta_t > 1.0e-6) {
      uint k = 0;
      delta_t_ik += delta_t;
      const Eigen::Vector3d phi_k = (imu_after[k].ang_vel - bg) * delta_t;
      const Eigen::Matrix3d C_k = C_mr;
      const Eigen::Matrix3d C_k_k_plus_1 = lgmath::so3::vec2rot(phi_k);

      r_rm_in_m += v_rm_in_m * delta_t + 0.5 * (C_mr * (imu_after[k].lin_acc - ba) + gravity_) * delta_t * delta_t;
      v_rm_in_m += (C_mr * (imu_after[k].lin_acc - ba) + gravity_) * delta_t;
      C_mr = C_mr * C_k_k_plus_1;

      IntegratedState state = IntegratedState(C_mr, r_rm_in_m, v_rm_in_m, imu_after[k].timestamp);
      if (compute_jacobians) {
        dpk_dri += dvk_dri * delta_t - 0.5 * C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * C_ik.transpose() * delta_t * delta_t;
        dpk_dbg += dvk_dbg * delta_t - 0.5 * C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * drk_dbg * delta_t * delta_t;
        dpk_dba += dvk_dba * delta_t - 0.5 * C_k * delta_t * delta_t;
        dvk_dri -= C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * C_ik.transpose() * delta_t;
        dvk_dbg -= C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * drk_dbg * delta_t;
        dvk_dba -= C_k * delta_t;
        // note: we multiply phi_k by (-1) to convert left Jacobian to right Jacobian
        drk_dbg = C_k_k_plus_1.transpose() * drk_dbg - lgmath::so3::vec2jac(-phi_k) * delta_t;
        // [dp / d delta x; dC / d delta x], x = (pos, rot, vel, b_a, b_g)
        Eigen::Matrix<double, 6, 15> jacobian = Eigen::Matrix<double, 6, 15>::Zero();
        // dp / d delta x 
        jacobian.block<3, 3>(0, 0) = C_i;
        jacobian.block<3, 3>(0, 3) = dpk_dri;
        jacobian.block<3, 3>(0, 6) = C_i * delta_t_ik;
        jacobian.block<3, 3>(0, 9) = dpk_dba;
        jacobian.block<3, 3>(0, 12) = dpk_dbg;
        // dC / d delta x
        C_ik = C_ik * C_k_k_plus_1;
        jacobian.block<3, 3>(3, 3) = C_ik.transpose();
        jacobian.block<3, 3>(3, 12) = drk_dbg;
        state.jacobian = jacobian;
      }
      integrated_states.push_back(state);
    }
    for (size_t k = 0; k < imu_after.size(); ++k) {
      double delta_t = 0;
      double pose_time = 0;
      if (k < imu_after.size() - 1) {
        delta_t = imu_after[k + 1].timestamp - imu_after[k].timestamp;
        pose_time = imu_after[k + 1].timestamp;
      } else if (k == imu_after.size() - 1) {
        delta_t = max_point_time_ - imu_after[k].timestamp;
        pose_time = max_point_time_;
      }
      assert(delta_t > 0);

      delta_t_ik += delta_t;
      const Eigen::Vector3d phi_k = (imu_after[k].ang_vel - bg) * delta_t;
      const Eigen::Matrix3d C_k = C_mr;
      const Eigen::Matrix3d C_k_k_plus_1 = lgmath::so3::vec2rot(phi_k);

      r_rm_in_m += v_rm_in_m * delta_t + 0.5 * (C_mr * (imu_after[k].lin_acc - ba) + gravity_) * delta_t * delta_t;
      v_rm_in_m += (C_mr * (imu_after[k].lin_acc - ba) + gravity_) * delta_t;
      C_mr = C_mr * C_k_k_plus_1;

      IntegratedState state = IntegratedState(C_mr, r_rm_in_m, v_rm_in_m, pose_time);
      if (compute_jacobians) {
        dpk_dri += dvk_dri * delta_t - 0.5 * C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * C_ik.transpose() * delta_t * delta_t;
        dpk_dbg += dvk_dbg * delta_t - 0.5 * C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * drk_dbg * delta_t * delta_t;
        dpk_dba += dvk_dba * delta_t - 0.5 * C_k * delta_t * delta_t;
        dvk_dri -= C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * C_ik.transpose() * delta_t;
        dvk_dbg -= C_k * lgmath::so3::hat(imu_after[k].lin_acc - ba) * drk_dbg * delta_t;
        dvk_dba -= C_k * delta_t;
        // note: we multiply phi_k by (-1) to convert left Jacobian to right Jacobian
        drk_dbg = C_k_k_plus_1.transpose() * drk_dbg - lgmath::so3::vec2jac(-phi_k) * delta_t;
        // [dp / d delta x; dC / d delta x], x = (pos, rot, vel, b_a, b_g)
        Eigen::Matrix<double, 6, 15> jacobian = Eigen::Matrix<double, 6, 15>::Zero();
        // dp / d delta x 
        jacobian.block<3, 3>(0, 0) = C_i;
        jacobian.block<3, 3>(0, 3) = dpk_dri;
        jacobian.block<3, 3>(0, 6) = C_i * delta_t_ik;
        jacobian.block<3, 3>(0, 9) = dpk_dba;
        jacobian.block<3, 3>(0, 12) = dpk_dbg;
        // dC / d delta x
        C_ik = C_ik * C_k_k_plus_1;
        jacobian.block<3, 3>(3, 3) = C_ik.transpose();
        jacobian.block<3, 3>(3, 12) = drk_dbg;
        state.jacobian = jacobian;
      }
      integrated_states.push_back(state);
    }
  }

  std::sort(integrated_states.begin(), integrated_states.end(), [](auto &left, auto &right) {
    return left.timestamp < right.timestamp;
  });

  return integrated_states;
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void P2PGlobalSuperCostTerm::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  
  using namespace steam::se3;
  using namespace steam::vspace;
  const auto T_ = transform_r_to_m_->forward();
  const auto v_ = v_m_to_r_in_m_->forward();
  const auto b_ = bias_->forward();

  const double rinv = 1.0 / options_.r_p2p;
  const double sqrt_rinv = sqrt(rinv);

  const std::vector<IntegratedState> states = integrate_(true);
  std::vector<double> pose_times;
  for (const IntegratedState &state : states) {
    pose_times.push_back(state.timestamp);
  }

  Eigen::Matrix<double, 15, 15> A = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 1> c = Eigen::Matrix<double, 15, 1>::Zero();

  // If some variables are not active? (simply don't use those parts
  // of the A, b to update hessian, grad at the end)
#pragma omp declare reduction(+ : Eigen::Matrix<double, 15, 15> : omp_out = \
                                  omp_out + omp_in)                         \
    initializer(omp_priv = Eigen::Matrix<double, 15, 15>::Zero())
#pragma omp declare reduction(+ : Eigen::Matrix<double, 15, 1> : omp_out = \
                                  omp_out + omp_in)                        \
    initializer(omp_priv = Eigen::Matrix<double, 15, 1>::Zero())
#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : A) \
    reduction(+ : c)
  for (int i = 0; i < (int)meas_times_.size(); ++i) {
    
    const double &ts = meas_times_[i];
    const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);
    // interpolation
    int start_index = 0, end_index = 0;
    for (size_t k = 0; k < pose_times.size(); ++k) {
      if (pose_times[k] > ts) {
        end_index = k;
        break;
      }
      start_index = k;
      end_index = k;
    }
    Eigen::Matrix3d C_mr = states[0].C_mr;
    Eigen::Vector3d r_rm_in_m = states[0].r_rm_in_m;
    double alpha = 0;
    if ((ts == pose_times[start_index]) || (start_index == end_index) || (pose_times[start_index] == pose_times[end_index])) {
      alpha = 0;
    } else if (ts == pose_times[end_index]) {
      alpha = 1;
    } else {
      alpha = (ts - pose_times[start_index]) / (pose_times[end_index] - pose_times[start_index]);
    }
    r_rm_in_m = alpha * states[end_index].r_rm_in_m + (1 - alpha) * states[start_index].r_rm_in_m;
    const Eigen::Vector3d phi = lgmath::so3::rot2vec(states[start_index].C_mr.transpose() * states[end_index].C_mr);
    C_mr = states[start_index].C_mr * lgmath::so3::vec2rot(alpha * phi);

    // interpolation Jacobians
    const Eigen::Matrix3d dr_dr1 = Eigen::Matrix3d::Identity() * (1 - alpha);
    const Eigen::Matrix3d dr_dr2 = Eigen::Matrix3d::Identity() * alpha;
    // note: we multiply the arguments by (-1) here in order to use the right-hand Jacobians of SO(3).
    const Eigen::Matrix3d a_jac = alpha * lgmath::so3::vec2jac(-alpha * phi) * lgmath::so3::vec2jacinv(-phi);
    const Eigen::Matrix3d dc_dc1 = Eigen::Matrix3d::Identity() - a_jac;
    const Eigen::Matrix3d dc_dc2 = a_jac;

    // measurement Jacobians
    Eigen::Matrix<double, 1, 3> de_dr = Eigen::Matrix<double, 1, 3>::Zero();
    Eigen::Matrix<double, 1, 3> de_dc = Eigen::Matrix<double, 1, 3>::Zero();
    double error = 0.0;

    for (const int &match_idx : bin_indices) {
      const auto &p2p_match = p2p_matches_.at(match_idx);
      const double raw_error =
          p2p_match.normal.transpose() *
          (p2p_match.reference - C_mr * p2p_match.query - r_rm_in_m);
      const double sqrt_w = sqrt(p2p_loss_func_->weight(fabs(raw_error)));
      error += sqrt_w * sqrt_rinv * raw_error;
      de_dr -= sqrt_w * sqrt_rinv * p2p_match.normal.transpose();
      de_dc += sqrt_w * sqrt_rinv * p2p_match.normal.transpose() * C_mr * lgmath::so3::hat(p2p_match.query);
    }
    // Get Jacobians of the bracketing states with respect to perturbations to the state variables
    const Eigen::Matrix<double, 3, 15> &dr1_dx = states[start_index].jacobian.block<3, 15>(0, 0);
    const Eigen::Matrix<double, 3, 15> &dc1_dx = states[start_index].jacobian.block<3, 15>(3, 0);
    const Eigen::Matrix<double, 3, 15> &dr2_dx = states[end_index].jacobian.block<3, 15>(0, 0);
    const Eigen::Matrix<double, 3, 15> &dc2_dx = states[end_index].jacobian.block<3, 15>(3, 0);
    const Eigen::Matrix<double, 1, 15> G = 
      de_dr * (dr_dr1 * dr1_dx + dr_dr2 * dr2_dx) +
      de_dc * (dc_dc1 * dc1_dx + dc_dc2 * dc2_dx);
    A += G.transpose() * G;
    c -= G.transpose() * error;
  }

  // Update hessian and grad for only the active variables
  std::vector<bool> active;
  active.push_back(transform_r_to_m_->active());
  active.push_back(v_m_to_r_in_m_->active());
  active.push_back(bias_->active());

  std::vector<StateKey> keys;

  if (active[0]) {
    const auto Tnode = std::static_pointer_cast<Node<PoseType>>(T_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    transform_r_to_m_->backward(lhs, Tnode, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[1]) {
    const auto vnode = std::static_pointer_cast<Node<VelType>>(v_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    v_m_to_r_in_m_->backward(lhs, vnode, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  } else {
    keys.push_back(-1);
  }
  if (active[2]) {
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

  std::vector<int> blk_indices = {0, 6, 9};
  std::vector<int> blk_sizes = {6, 3, 6};

  for (int i = 0; i < 3; ++i) {
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

    for (int j = i; j < 3; ++j) {
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
            return A.block<3, 3>(blk_indices[i], blk_indices[j]).transpose();
          } else if (blk_sizes[i] == 3 && blk_sizes[j] == 6) {
            return A.block<3, 6>(blk_indices[i], blk_indices[j]).transpose();
          } else if (blk_sizes[i] == 6 && blk_sizes[j] == 3) {
            return A.block<6, 3>(blk_indices[i], blk_indices[j]).transpose();
          } else if (blk_sizes[i] == 6 && blk_sizes[j] == 6) {
            return A.block<6, 6>(blk_indices[i], blk_indices[j]).transpose();
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
