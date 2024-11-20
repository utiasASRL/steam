#pragma once

#include "steam/evaluable/se3/se3_state_var.hpp"
#include "steam/evaluable/state_var.hpp"
#include "steam/evaluable/vspace/vspace_state_var.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/cost_term/p2p_super_cost_term.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/problem.hpp"
#include "steam/trajectory/const_acc/interface.hpp"
#include "steam/trajectory/time.hpp"

#include <iostream>

namespace steam {

class P2PDopplerCASuperCostTerm : public BaseCostTerm {
 public:
  enum class LOSS_FUNC { L2, DCS, CAUCHY, GM, HUBER };

  struct Options {
    int num_threads = 1;
    LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
    double p2p_loss_sigma = 0.1;
    Eigen::Matrix4d T_sr = Eigen::Matrix4d::Identity();
    double beta = 0.0535;
  };

  using Ptr = std::shared_ptr<P2PDopplerCASuperCostTerm>;
  using ConstPtr = std::shared_ptr<const P2PDopplerCASuperCostTerm>;

  using PoseType = lgmath::se3::Transformation;
  using VelType = Eigen::Matrix<double, 6, 1>;
  using AccType = Eigen::Matrix<double, 6, 1>;

  using Interface = steam::traj::const_acc::Interface;

  using Variable = steam::traj::const_acc::Variable;

  using Time = steam::traj::Time;

  using Matrix18d = Eigen::Matrix<double, 18, 18>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  static Ptr MakeShared(const Interface::ConstPtr &interface, const Time time1,
                        const Time time2, const Options &options);

  P2PDopplerCASuperCostTerm(const Interface::ConstPtr &interface,
                            const Time time1, const Time time2,
                            const Options &options)
      : interface_(interface),
        time1_(time1),
        time2_(time2),
        options_(options),
        knot1_(interface_->get(time1)),
        knot2_(interface_->get(time2)) {
    const double T = (knot2_->time() - knot1_->time()).seconds();
    const Eigen::Matrix<double, 6, 1> ones =
        Eigen::Matrix<double, 6, 1>::Ones();
    Qinv_T_ = interface_->getQinvPublic(T, ones);
    Tran_T_ = interface_->getTranPublic(T);

    p2p_loss_func_ = [this]() -> BaseLossFunc::Ptr {
      switch (options_.p2p_loss_func) {
        case LOSS_FUNC::L2:
          return L2LossFunc::MakeShared();
        case LOSS_FUNC::DCS:
          return DcsLossFunc::MakeShared(options_.p2p_loss_sigma);
        case LOSS_FUNC::CAUCHY:
          return CauchyLossFunc::MakeShared(options_.p2p_loss_sigma);
        case LOSS_FUNC::GM:
          return GemanMcClureLossFunc::MakeShared(options_.p2p_loss_sigma);
        case LOSS_FUNC::HUBER:
          return HuberLossFunc::MakeShared(options_.p2p_loss_sigma);
        default:
          return nullptr;
      }
      return nullptr;
    }();

    T_rs_ = options_.T_sr.inverse();
    Ad_T_sr_ = lgmath::se3::tranAd(options_.T_sr);
  }

  /** \brief Compute the cost to the objective function */
  double cost() const override;

  /** \brief Get keys of variables related to this cost term */
  void getRelatedVarKeys(KeySet &keys) const override;

  void initP2PMatches();

  void emplace_back(P2PMatch &p2p_match) {
    p2p_matches_.emplace_back(p2p_match);
  }

  void clear() { p2p_matches_.clear(); }
  void reserve(unsigned int N) { p2p_matches_.reserve(N); }

  std::vector<P2PMatch> &get() { return p2p_matches_; }

  /**
   * \brief Add the contribution of this cost term to the left-hand (Hessian)
   * and right-hand (gradient vector) sides of the Gauss-Newton system of
   * equations.
   */
  void buildGaussNewtonTerms(const StateVector &state_vec,
                             BlockSparseMatrix *approximate_hessian,
                             BlockVector *gradient_vector) const override;

 private:
  const Interface::ConstPtr interface_;
  const Time time1_;
  const Time time2_;
  const Options options_;
  const Variable::ConstPtr knot1_;
  const Variable::ConstPtr knot2_;
  Matrix18d Qinv_T_ = Matrix18d::Identity();
  Matrix18d Tran_T_ = Matrix18d::Identity();
  std::map<double, std::pair<Matrix18d, Matrix18d>> interp_mats_;

  std::vector<P2PMatch> p2p_matches_;
  std::map<double, std::vector<int>> p2p_match_bins_;
  std::vector<double> meas_times_;

  BaseLossFunc::Ptr p2p_loss_func_ = L2LossFunc::MakeShared();

  void initialize_interp_matrices_();

  Eigen::Matrix4d T_rs_ = Eigen::Matrix4d::Identity();
  Eigen::Matrix<double, 6, 6> Ad_T_sr_ =
      Eigen::Matrix<double, 6, 6>::Identity();
};

}  // namespace steam
