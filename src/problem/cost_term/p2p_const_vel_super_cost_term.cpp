#include <iostream>

#include "steam/problem/cost_term/p2p_const_vel_super_cost_term.hpp"

namespace steam {

P2PCVSuperCostTerm::Ptr P2PCVSuperCostTerm::MakeShared(
    const Interface::ConstPtr &interface, const Time &time1, const Time &time2,
    const Options &options) {
  return std::make_shared<P2PCVSuperCostTerm>(interface, time1, time2, options);
}

/** \brief Compute the cost to the objective function */
double P2PCVSuperCostTerm::cost() const {
  double cost = 0;
  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto J_21_inv_w2 = J_21_inv * w2;

#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : cost)
  for (unsigned int i = 0; i < meas_times_.size(); ++i) {
    const double &ts = meas_times_[i];
    const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);

    // pose interpolation
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv_w2;
    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;
    const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

    for (const int &match_idx : bin_indices) {
      const auto &p2p_match = p2p_matches_.at(match_idx);
      const double raw_error =
          p2p_match.normal.transpose() *
          (p2p_match.reference - T_mr.block<3, 3>(0, 0) * p2p_match.query -
           T_mr.block<3, 1>(0, 3));
      cost += p2p_loss_func_->cost(fabs(raw_error));
    }
  }
  return cost;
}

/** \brief Get keys of variables related to this cost term */
void P2PCVSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
}

void P2PCVSuperCostTerm::initP2PMatches() {
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
  initialize_interp_matrices_();
}

void P2PCVSuperCostTerm::initialize_interp_matrices_() {
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
  for (const double &time : meas_times_) {
    if (interp_mats_.find(time) == interp_mats_.end()) {
      // Get Lambda, Omega for this time
      const double tau = time - time1_.seconds();
      const double kappa = knot2_->time().seconds() - time;
      const Matrix12d Q_tau = steam::traj::const_vel::getQ(tau, ones);
      const Matrix12d Tran_kappa = steam::traj::const_vel::getTran(kappa);
      const Matrix12d Tran_tau = steam::traj::const_vel::getTran(tau);
      const Matrix12d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
      const Matrix12d lambda = (Tran_tau - omega * Tran_T_);
      interp_mats_.emplace(time, std::make_pair(omega, lambda));
    }
  }
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void P2PCVSuperCostTerm::buildGaussNewtonTerms(
    const StateVector &state_vec, BlockSparseMatrix *approximate_hessian,
    BlockVector *gradient_vector) const {
  using namespace steam::se3;
  using namespace steam::vspace;
  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();

  Eigen::Matrix<double, 24, 24> A = Eigen::Matrix<double, 24, 24>::Zero();
  Eigen::Matrix<double, 24, 1> b = Eigen::Matrix<double, 24, 1>::Zero();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const auto Ad_T_21 = lgmath::se3::tranAd(T_21.matrix());
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto w2_j_21_inv = 0.5 * lgmath::se3::curlyhat(w2) * J_21_inv;
  const auto J_21_inv_w2 = J_21_inv * w2;

  // If some variables are not active? (simply don't use those parts
  // of the A, b to update hessian, grad at the end)
#pragma omp declare reduction(+ : Eigen::Matrix<double, 24, 24> : omp_out = \
                                  omp_out + omp_in)                         \
    initializer(omp_priv = Eigen::Matrix<double, 24, 24>::Zero())
#pragma omp declare reduction(+ : Eigen::Matrix<double, 24, 1> : omp_out = \
                                  omp_out + omp_in)                        \
    initializer(omp_priv = Eigen::Matrix<double, 24, 1>::Zero())
#pragma omp parallel for num_threads(options_.num_threads) reduction(+ : A) \
    reduction(+ : b)
  for (int i = 0; i < (int)meas_times_.size(); ++i) {
    const double &ts = meas_times_[i];
    const std::vector<int> &bin_indices = p2p_match_bins_.at(ts);

    // pose interpolation
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv_w2;
    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;
    const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

    // pose interpolation Jacobians
    Eigen::Matrix<double, 6, 24> interp_jac =
        Eigen::Matrix<double, 6, 24>::Zero();

    const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);

    const Eigen::Matrix<double, 6, 6> w =
        J_i1 * (omega.block<6, 6>(0, 0) * J_21_inv +
                omega.block<6, 6>(0, 6) * w2_j_21_inv);

    interp_jac.block<6, 6>(0, 0) = -w * Ad_T_21 + T_i1.adjoint();    // T1
    interp_jac.block<6, 6>(0, 6) = lambda.block<6, 6>(0, 6) * J_i1;  // w1
    interp_jac.block<6, 6>(0, 12) = w;                               // T2
    interp_jac.block<6, 6>(0, 18) =
        omega.block<6, 6>(0, 6) * J_i1 * J_21_inv;  // w2

    // measurement Jacobians
    Eigen::Matrix<double, 1, 6> Gmeas = Eigen::Matrix<double, 1, 6>::Zero();
    double error = 0.0;

    for (const int &match_idx : bin_indices) {
      const auto &p2p_match = p2p_matches_.at(match_idx);
      const double raw_error =
          p2p_match.normal.transpose() *
          (p2p_match.reference - T_mr.block<3, 3>(0, 0) * p2p_match.query -
           T_mr.block<3, 1>(0, 3));
      const double sqrt_w = sqrt(p2p_loss_func_->weight(fabs(raw_error)));
      error += sqrt_w * raw_error;
      Gmeas +=
          sqrt_w * p2p_match.normal.transpose() *
          (T_mr * lgmath::se3::point2fs(p2p_match.query)).block<3, 6>(0, 0);
    }
    const Eigen::Matrix<double, 1, 24> G = Gmeas * interp_jac;
    A += G.transpose() * G;
    b += (-1) * G.transpose() * error;
  }

  // Update hessian and grad for only the active variables
  std::vector<StateKey> keys;
  std::vector<bool> active;
  {
    const auto T1node = std::static_pointer_cast<Node<PoseType>>(T1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot1_->pose()->backward(lhs, T1node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  }
  {
    const auto w1node = std::static_pointer_cast<Node<VelType>>(w1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot1_->velocity()->backward(lhs, w1node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  }
  {
    const auto T2node = std::static_pointer_cast<Node<PoseType>>(T2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot2_->pose()->backward(lhs, T2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  }
  {
    const auto w2node = std::static_pointer_cast<Node<VelType>>(w2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot2_->velocity()->backward(lhs, w2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  }
  // std::cout << "keys:";
  // for (auto &key : keys) {
  //   std::cout << state_vec.getStateBlockIndex(key) << " ";
  // }
  // std::cout << std::endl;
  active.push_back(knot1_->pose()->active());
  active.push_back(knot1_->velocity()->active());
  active.push_back(knot2_->pose()->active());
  active.push_back(knot2_->velocity()->active());

  for (int i = 0; i < 4; ++i) {
    if (!active[i]) continue;
    // Get the key and state range affected
    const auto &key1 = keys[i];
    unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

    // Calculate terms needed to update the right-hand-side
    Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

    // Update the right-hand side (thread critical)
#pragma omp critical(b_update)
    { gradient_vector->mapAt(blkIdx1) += newGradTerm; }

    for (int j = i; j < 4; ++j) {
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
