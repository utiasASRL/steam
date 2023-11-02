#pragma once

#include "steam/evaluable/p2p/p2p_error_evaluator.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/problem.hpp"
#include "steam/trajectory/const_acc/interface.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {

class P2PSuperCostTerm : public BaseCostTerm {
 public:
  enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options {
    // bool bias_const_over_window = false;
    // bool estimate_T_mi = true;
    // bool T_mi_const_over_window = false;
    // bool T_mi_init_only = false;
    // Eigen::Matrix<double, 6, 1> Qc_bias = Eigen::Matrix<double, 6,
    // 1>::Ones(); Eigen::Matrix<double, 6, 1> Qc_T_mi = Eigen::Matrix<double,
    // 6, 1>::Ones(); Eigen::Matrix<double, 3, 1> r_imu_acc =
    // Eigen::Matrix<double, 3, 1>::Zero(); Eigen::Matrix<double, 3, 1>
    // r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
    int num_threads = 1;
    // LOSS_FUNC accel_loss_func = LOSS_FUNC::L2;
    // LOSS_FUNC gyro_loss_func = LOSS_FUNC::L2;
    LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
    // double accel_loss_sigma = 1.0;
    // double gyro_loss_sigma = 1.0;
    double p2p_loss_sigma = 0.1;
    Eigen::Matrix4d T_sr = Eigen::Matrix4d::Identity();
  };

  using Ptr = std::shared_ptr<P2PSuperCostTerm>;
  using ConstPtr = std::shared_ptr<const P2PSuperCostTerm>;

  using PoseType = lgmath::se3::Transformation;

  using Interface = steam::traj::const_acc::Interface;

  using Variable = steam::traj::const_acc::Variable;

  using Time = steam::traj::Time;

  using Matrix18d = Eigen::Matrix<double, 18, 18>;
  using Matrix18d = Eigen::Matrix<double, 6, 6>;

  static Ptr MakeShared(const Interface::ConstPtr &interface, const Time &time1,
                        const Time &time2, Options options);
  P2PSuperCostTerm(const Interface::ConstPtr &interface, const Time &time1,
                   const Time &time2, Options options);

  /** \brief Compute the cost to the objective function */
  double cost() const override;

  /** \brief Get keys of variables related to this cost term */
  void getRelatedVarKeys(KeySet &keys) const override;

  void setP2PMatches(const std::vector<P2PMatch> &p2p_matches);

  void clearP2PMatches() { p2p_matches_.clear(); }

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
  const Time &time1_;
  const Time &time2_;
  Variable::ConstPtr knot1_;
  Variable::ConstPtr knot2_;
  Options options_;
  Matrix18d Qinv_T_ = Matrix18d::Identity();
  Matrix18d Tran_T_ = Matrix18d::Identity();
  // std::set<Time> meas_times_;
  std::map<Time, std::pair<Matrix18d, Matrix18d>> interp_mats_;

  std::vector<P2PMatch> p2p_matches_;

  BaseLossFunc::Ptr p2p_loss_func_ = L2LossFunc::MakeShared();

  void getMotionPriorJacobians_(const lgmath::se3::Transformation &T1,
                                const lgmath::se3::Transformation &T2,
                                const Eigen::Matrix<double, 6, 1> &w2,
                                const Eigen::Matrix<double, 6, 1> &dw2,
                                const Eigen::Matrix<double, 18, 18> &Phi,
                                Eigen::Matrix<double, 18, 18> &F,
                                Eigen::Matrix<double, 18, 18> &E) const;

  void initialize_interp_matrices_();
};

}  // namespace steam
