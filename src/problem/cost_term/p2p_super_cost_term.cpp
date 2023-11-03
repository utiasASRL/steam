#include "steam/problem/cost_term/p2p_super_cost_term.hpp"

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
  knot1_ = interface_->get(time1_);
  knot2_ = interface_->get(time2_);

  const double T = (knot2_->time() - knot1_->time()).seconds();
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
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
double P2PSuperCostTerm::cost() const {
  const auto T1_ = knot1_->pose()->forward();
  const auto w1_ = knot1_->velocity()->forward();
  const auto dw1_ = knot1_->acceleration()->forward();
  const auto T2_ = knot2_->pose()->forward();
  const auto w2_ = knot2_->velocity()->forward();
  const auto dw2_ = knot2_->acceleration()->forward();

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto dw1 = dw1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();
  const auto dw2 = dw2_->value();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto J_21_inv_w2 = J_21_inv * w2;
  const auto J_21_inv_curl_dw2 =
      (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

  double cost = 0;
  // #pragma omp parallel for num_threads(options_.num_threads) reduction(+ :
  // cost)
  for (auto it = p2p_match_bins_.begin(); it != p2p_match_bins_.end(); it++) {
    const double &ts = it->first;
    const std::vector<int> &bin_indices = it->second;

    // pose interpolation
    // const std::pair<Matrix18d, Matrix18d> &omega_lambda = interp_mats_[ts];
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
        omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv_w2 +
        omega.block<6, 6>(0, 12) * J_21_inv_curl_dw2;
    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;
    const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

    for (const int &match_idx : bin_indices) {
      const auto &p2p_match = p2p_matches_->at(match_idx);
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
void P2PSuperCostTerm::getRelatedVarKeys(KeySet &keys) const {
  knot1_->pose()->getRelatedVarKeys(keys);
  knot1_->velocity()->getRelatedVarKeys(keys);
  knot1_->acceleration()->getRelatedVarKeys(keys);
  knot2_->pose()->getRelatedVarKeys(keys);
  knot2_->velocity()->getRelatedVarKeys(keys);
  knot2_->acceleration()->getRelatedVarKeys(keys);
}

void P2PSuperCostTerm::setP2PMatches(std::vector<P2PMatch> *p2p_matches) {
  p2p_matches_ = p2p_matches;
  p2p_match_bins_.clear();
  for (unsigned int i = 0; i < p2p_matches->size(); ++i) {
    const auto &p2p_match = p2p_matches->at(i);
    const auto &timestamp = p2p_match.timestamp;
    if (p2p_match_bins_.find(timestamp) == p2p_match_bins_.end()) {
      p2p_match_bins_[timestamp] = {i};
    } else {
      p2p_match_bins_[timestamp].push_back(i);
    }
  }
  initialize_interp_matrices_();
}

void P2PSuperCostTerm::initialize_interp_matrices_() {
  const Eigen::Matrix<double, 6, 1> ones = Eigen::Matrix<double, 6, 1>::Ones();
  for (auto it = p2p_match_bins_.begin(); it != p2p_match_bins_.end(); ++it) {
    const auto &time = it->first;
    if (interp_mats_.find(time) == interp_mats_.end()) {
      // Get Lambda, Omega for this time
      const double tau = time - time1_.seconds();
      const double kappa = time2_.seconds() - time;
      const Matrix18d Q_tau = interface_->getQPublic(tau, ones);
      const Matrix18d Tran_kappa = interface_->getTranPublic(kappa);
      const Matrix18d Tran_tau = interface_->getTranPublic(tau);
      const Matrix18d omega = (Q_tau * Tran_kappa.transpose() * Qinv_T_);
      const Matrix18d lambda = (Tran_tau - omega * Tran_T_);
      interp_mats_.emplace(time, std::make_pair(omega, lambda));
    }
  }
}

/**
 * \brief Add the contribution of this cost term to the left-hand (Hessian)
 * and right-hand (gradient vector) sides of the Gauss-Newton system of
 * equations.
 */
void P2PSuperCostTerm::buildGaussNewtonTerms(
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

  const auto T1 = T1_->value();
  const auto w1 = w1_->value();
  const auto dw1 = dw1_->value();
  const auto T2 = T2_->value();
  const auto w2 = w2_->value();
  const auto dw2 = dw2_->value();

  const auto xi_21 = (T2 / T1).vec();
  const lgmath::se3::Transformation T_21(xi_21);
  const Eigen::Matrix<double, 6, 6> J_21_inv = lgmath::se3::vec2jacinv(xi_21);
  const auto J_21_inv_w2 = J_21_inv * w2;
  const auto J_21_inv_curl_dw2 =
      (-0.5 * lgmath::se3::curlyhat(J_21_inv * w2) * w2 + J_21_inv * dw2);

  // todo: what if some variables are not active? (simply don't use those parts
  // of the A, b to update hessian, grad at the end)
  Eigen::Matrix<double, 36, 36> A = Eigen::Matrix<double, 36, 36>::Zero();
  Eigen::Matrix<double, 36, 1> b = Eigen::Matrix<double, 36, 1>::Zero();
#pragma omp declare reduction( \
        merge_A : Eigen::Matrix<double, 36, 36> : omp_out += omp_in)
#pragma omp declare reduction( \
        merge_b : Eigen::Matrix<double, 36, 1> : omp_out += omp_in)
#pragma omp parallel for num_threads(options_.num_threads) \
    reduction(merge_A : A) reduction(merge_b : b)
  for (auto it = p2p_match_bins_.begin(); it != p2p_match_bins_.end(); it++) {
    const double &ts = it->first;
    const std::vector<int> &bin_indices = it->second;

    // pose interpolation
    const auto &omega = interp_mats_.at(ts).first;
    const auto &lambda = interp_mats_.at(ts).second;
    const Eigen::Matrix<double, 6, 1> xi_i1 =
        lambda.block<6, 6>(0, 6) * w1 + lambda.block<6, 6>(0, 12) * dw1 +
        omega.block<6, 6>(0, 0) * xi_21 +
        omega.block<6, 6>(0, 6) * J_21_inv_w2 +
        omega.block<6, 6>(0, 12) * J_21_inv_curl_dw2;
    const lgmath::se3::Transformation T_i1(xi_i1);
    const lgmath::se3::Transformation T_i0 = T_i1 * T1;
    const Eigen::Matrix4d T_mr = T_i0.inverse().matrix();

    // pose interpolation Jacobian
    Eigen::Matrix<double, 6, 36> interp_jac =
        Eigen::Matrix<double, 6, 36>::Zero();

    const Eigen::Matrix<double, 6, 6> J_i1 = lgmath::se3::vec2jac(xi_i1);

    const Eigen::Matrix<double, 6, 6> w =
        J_i1 *
        (omega.block<6, 6>(0, 0) * Eigen::Matrix<double, 6, 6>::Identity() +
         omega.block<6, 6>(0, 6) * 0.5 * lgmath::se3::curlyhat(w2) +
         omega.block<6, 6>(0, 12) * 0.25 * lgmath::se3::curlyhat(w2) *
             lgmath::se3::curlyhat(w2) +
         omega.block<6, 6>(0, 12) * 0.5 * lgmath::se3::curlyhat(dw2)) *
        J_21_inv;

    interp_jac.block<6, 6>(0, 0) = -w * T_21.adjoint() + T_i1.adjoint();  // T1
    interp_jac.block<6, 6>(0, 6) = lambda.block<6, 6>(0, 6) * J_i1;       // w1
    interp_jac.block<6, 6>(0, 12) = lambda.block<6, 6>(0, 12) * J_i1;     // dw1
    interp_jac.block<6, 6>(0, 18) = w;                                    // T2
    interp_jac.block<6, 6>(0, 24) =
        omega.block<6, 6>(0, 6) * J_i1 * J_21_inv +
        omega.block<6, 6>(0, 12) * -0.5 * J_i1 *
            (lgmath::se3::curlyhat(J_21_inv * w2) -
             lgmath::se3::curlyhat(w2) * J_21_inv);  // w2
    interp_jac.block<6, 6>(0, 30) =
        omega.block<6, 6>(0, 12) * J_i1 * J_21_inv;  // dw2

    // get measurement Jacobians
    Eigen::Matrix<double, 1, 6> Gmeas = Eigen::Matrix<double, 1, 6>::Zero();
    double error;
    for (const int &match_idx : bin_indices) {
      const auto &p2p_match = p2p_matches_->at(match_idx);
      const double raw_error =
          p2p_match.normal.transpose() *
          (p2p_match.reference - T_mr.block<3, 3>(0, 0) * p2p_match.query -
           T_mr.block<3, 1>(0, 3));
      const double sqrt_w = sqrt(p2p_loss_func_->weight(fabs(raw_error)));
      // weight meas jacs and errors
      error += sqrt_w * raw_error;
      Gmeas +=
          sqrt_w * p2p_match.normal.transpose() *
          (T_mr * lgmath::se3::point2fs(p2p_match.query)).block<3, 6>(0, 0);
    }
    const Eigen::Matrix<double, 1, 36> G = Gmeas * interp_jac;
    A += G.transpose() * G;
    b += (-1) * G.transpose() * error;
  }

  // Update hessian and grad for only the active variables
  // std::vector<StateVar::Ptr> vars;
  // vars.emplace_back(std::static_pointer_cast<StateVar::Ptr>(knot1_->pose()));

  // vars.emplace_back(
  //     std::static_pointer_cast<StateVar::Ptr>(knot1_->velocity()));
  // vars.emplace_back(
  //     std::static_pointer_cast<StateVar::Ptr>(knot1_->acceleration()));
  // vars.emplace_back(std::static_pointer_cast<StateVar::Ptr>(knot2_->pose()));
  // vars.emplace_back(
  //     std::static_pointer_cast<StateVar::Ptr>(knot2_->velocity()));
  // vars.emplace_back(
  //     std::static_pointer_cast<StateVar::Ptr>(knot2_->acceleration()));

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
    const auto dw1node = std::static_pointer_cast<Node<AccType>>(dw1_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot1_->acceleration()->backward(lhs, dw1node, jacs);
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
  {
    const auto dw2node = std::static_pointer_cast<Node<AccType>>(dw2_);
    Jacobians jacs;
    Eigen::Matrix<double, 1, 1> lhs = Eigen::Matrix<double, 1, 1>::Zero();
    knot2_->acceleration()->backward(lhs, dw2node, jacs);
    const auto jacmap = jacs.get();
    assert(jacmap.size() == 1);
    for (auto it = jacmap.begin(); it != jacmap.end(); it++) {
      keys.push_back(it->first);
    }
  }
  active.push_back(knot1_->pose()->active());
  active.push_back(knot1_->velocity()->active());
  active.push_back(knot1_->acceleration()->active());
  active.push_back(knot2_->pose()->active());
  active.push_back(knot2_->velocity()->active());
  active.push_back(knot2_->acceleration()->active());

  for (int i = 0; i < 6; ++i) {
    if (!active[i]) continue;
    // Get the key and state range affected
    // const auto &key1 = vars[i]->key();
    const auto &key1 = keys[i];
    unsigned int blkIdx1 = state_vec.getStateBlockIndex(key1);

    // Calculate terms needed to update the right-hand-side
    Eigen::MatrixXd newGradTerm = b.block<6, 1>(i * 6, 0);

    // Update the right-hand side (thread critical)

#pragma omp critical(b_update)
    gradient_vector->mapAt(blkIdx1) += newGradTerm;

    for (int j = i; j < 6; ++j) {
      if (!active[j]) continue;
      // Get the key and state range affected
      // const auto &key2 = vars[j]->key();
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
