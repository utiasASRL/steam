#pragma once

#include <steam/trajectory/const_vel/helper.hpp>
#include "steam/evaluable/se3/se3_state_var.hpp"
#include "steam/evaluable/state_var.hpp"
#include "steam/evaluable/vspace/vspace_state_var.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/cost_term/p2p_super_cost_term.hpp"
#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/problem.hpp"
#include "steam/trajectory/const_vel/interface.hpp"
#include "steam/trajectory/time.hpp"

#include <iostream>

namespace steam {

struct IntegratedState {
  Eigen::Matrix3d C_mr;
  Eigen::Vector3d r_rm_in_m;
  Eigen::Vector3d v_rm_in_m;
  double timestamp;
  Eigen::Matrix<double, 6, 15> jacobian;  // [dp / d delta x; dC / d delta x], x = (pos, rot, vel, b_a, b_g)

  IntegratedState(Eigen::Matrix3d C_mr_, Eigen::Vector3d r_rm_in_m_, Eigen::Vector3d v_rm_in_m_, double timestamp_) : C_mr(C_mr_), r_rm_in_m(r_rm_in_m_), v_rm_in_m(v_rm_in_m_), timestamp(timestamp_) {
    jacobian = Eigen::Matrix<double, 6, 15>::Zero();
  }
  IntegratedState() {}
};

class P2PGlobalSuperCostTerm : public BaseCostTerm {
 public:
  enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options {
    int num_threads = 1;
    LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
    double p2p_loss_sigma = 0.1;
    double r_p2p = 1.0;
    Eigen::Matrix<double, 3, 1> gravity = {0, 0, -9.8042};
  };

  using Ptr = std::shared_ptr<P2PGlobalSuperCostTerm>;
  using ConstPtr = std::shared_ptr<const P2PGlobalSuperCostTerm>;

  using PoseType = lgmath::se3::Transformation;
  using VelType = Eigen::Matrix<double, 3, 1>;
  using BiasType = Eigen::Matrix<double, 6, 1>;

  using Time = steam::traj::Time;

  static Ptr MakeShared(
    const Time time,
    const Evaluable<PoseType>::ConstPtr &transform_r_to_m,
    const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m,
    const Evaluable<BiasType>::ConstPtr &bias,
    const Options &options,
    const std::vector<IMUData> &imu_data_vec);

  P2PGlobalSuperCostTerm(
    const Time time,
    const Evaluable<PoseType>::ConstPtr &transform_r_to_m,
    const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m,
    const Evaluable<BiasType>::ConstPtr &bias,
    const Options &options,
    const std::vector<IMUData> &imu_data_vec)
      : time_(time),
        transform_r_to_m_(transform_r_to_m),
        v_m_to_r_in_m_(v_m_to_r_in_m),
        bias_(bias),
        options_(options),
        curr_time_(time.seconds()) {

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
        default:
          return nullptr;
      }
      return nullptr;
    }();
    gravity_ = options_.gravity;
    for (auto imu_data : imu_data_vec) {
      imu_data_vec_.push_back(imu_data);
    }
    for (auto imu_data : imu_data_vec_) {
      if (imu_data.timestamp < curr_time_) {
        imu_before.push_back(imu_data);
      } else {
        imu_after.push_back(imu_data);
      }
    }
    if (imu_before.size() > 0) {
      std::reverse(imu_before.begin(), imu_before.end());
    }
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
  
  std::vector<IntegratedState> integrate_(bool compute_jacobians) const;

  void set_min_time(double min_time) {
    min_point_time_ = min_time;
  }

  void set_max_time(double max_time) {
    max_point_time_ = max_time;
  }

 private:
  const Time time_;
  const Evaluable<PoseType>::ConstPtr transform_r_to_m_;
  const Evaluable<VelType>::ConstPtr v_m_to_r_in_m_;
  const Evaluable<BiasType>::ConstPtr bias_;
  const Options options_;
  const double curr_time_;

  std::vector<steam::IMUData> imu_data_vec_;
  std::vector<steam::IMUData> imu_before;
  std::vector<steam::IMUData> imu_after;

  std::vector<P2PMatch> p2p_matches_;
  std::map<double, std::vector<int>> p2p_match_bins_;
  std::vector<double> meas_times_;
  double min_point_time_ = 0;
  double max_point_time_ = 0;

  BaseLossFunc::Ptr p2p_loss_func_ = L2LossFunc::MakeShared();
  Eigen::Vector3d gravity_ = {0, 0, -9.8042};

  // void initialize_interp_matrices_();
};

}  // namespace steam
