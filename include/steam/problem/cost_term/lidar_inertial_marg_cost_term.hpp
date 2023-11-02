#pragma once

#include "steam/evaluable/imu/acc_error_evaluator.hpp"
#include "steam/evaluable/imu/bias_interpolator.hpp"
#include "steam/evaluable/imu/gyro_error_evaluator.hpp"
#include "steam/evaluable/p2p/p2p_error_evaluator.hpp"
#include "steam/evaluable/se3/pose_interpolator.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/problem.hpp"
#include "steam/trajectory/const_acc/interface.hpp"
#include "steam/trajectory/time.hpp"

namespace steam {

class LidarInertialMarginalizedCostTerm : public BaseCostTerm {
 public:
  enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options {
    bool bias_const_over_window = false;
    bool estimate_T_mi = true;
    bool T_mi_const_over_window = false;
    bool T_mi_init_only = false;
    Eigen::Matrix<double, 6, 1> Qc_bias = Eigen::Matrix<double, 6, 1>::Ones();
    Eigen::Matrix<double, 6, 1> Qc_T_mi = Eigen::Matrix<double, 6, 1>::Ones();
    Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
    int num_threads = 1;
    LOSS_FUNC accel_loss_func = LOSS_FUNC::L2;
    LOSS_FUNC gyro_loss_func = LOSS_FUNC::L2;
    LOSS_FUNC p2p_loss_func = LOSS_FUNC::CAUCHY;
    double accel_loss_sigma = 1.0;
    double gyro_loss_sigma = 1.0;
    double p2p_loss_sigma = 0.1;
  };

  using Ptr = std::shared_ptr<LidarInertialMarginalizedCostTerm>;
  using ConstPtr = std::shared_ptr<const LidarInertialMarginalizedCostTerm>;

  using BiasType = Eigen::Matrix<double, 6, 1>;
  using PoseType = lgmath::se3::Transformation;

  using Interface = steam::traj::const_acc::Interface;
  using AccelerationErrorEvaluator = steam::imu::AccelerationErrorEvaluator;
  using P2PErrorEvaluator = steam::p2p::P2PErrorEvaluator;
  using GyroErrorEvaluator = steam::imu::GyroErrorEvaluator;

  using Variable = steam::traj::const_acc::Variable;

  using Time = steam::traj::Time;

  static Ptr MakeShared(const Interface::ConstPtr &interface, const Time &time1,
                        const Evaluable<BiasType>::ConstPtr &bias1,
                        const Evaluable<PoseType>::ConstPtr &T_mi_1,
                        const Time &time2,
                        const Evaluable<BiasType>::ConstPtr &bias2,
                        const Evaluable<PoseType>::ConstPtr &T_mi_2,
                        Options options);
  LidarInertialMarginalizedCostTerm(const Interface::ConstPtr &interface,
                                    const Time &time1,
                                    const Evaluable<BiasType>::ConstPtr &bias1,
                                    const Evaluable<PoseType>::ConstPtr &T_mi_1,
                                    const Time &time2,
                                    const Evaluable<BiasType>::ConstPtr &bias2,
                                    const Evaluable<PoseType>::ConstPtr &T_mi_2,
                                    Options options);

  /** \brief Compute the cost to the objective function */
  double cost() const override;

  /** \brief Get keys of variables related to this cost term */
  void getRelatedVarKeys(KeySet &keys) const override;

  void addAccelCostTerms(
      const std::vector<AccelerationErrorEvaluator::ConstPtr> &accel_err_vec);

  void addGyroCostTerms(
      const std::vector<GyroErrorEvaluator::ConstPtr> &gyro_err_vec);

  void addP2PCostTerms(
      const std::vector<P2PErrorEvaluator::ConstPtr> &p2p_err_vec,
      const std::vector<Eigen::Matrix<double, 3, 3>> &W_vec);

  void clearAccelCostTerms() { accel_cost_terms_.clear(); }
  void clearGyroCostTerms() { gyro_cost_terms_.clear(); }
  void clearP2PCostTerms() { p2p_cost_terms_.clear(); }

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
  const Evaluable<BiasType>::ConstPtr bias1_;
  const Evaluable<PoseType>::ConstPtr &T_mi_1_;
  const Time &time2_;
  const Evaluable<BiasType>::ConstPtr &bias2_;
  const Evaluable<PoseType>::ConstPtr &T_mi_2_;
  Variable::ConstPtr knot1_;
  Variable::ConstPtr knot2_;
  Options options_;
  std::set<Time> meas_times_;

  std::vector<AccelerationErrorEvaluator::ConstPtr> accel_err_vec_;
  std::vector<GyroErrorEvaluator::ConstPtr> gyro_err_vec_;
  std::vector<P2PErrorEvaluator::ConstPtr> p2p_err_vec_;
  std::vector<StaticNoiseModel::ConstPtr> p2p_noise_models_;

  BaseLossFunc::Ptr acc_loss_func_ = L2LossFunc::MakeShared();
  BaseLossFunc::Ptr gyro_loss_func_ = L2LossFunc::MakeShared();
  BaseLossFunc::Ptr p2p_loss_func_ = L2LossFunc::MakeShared();

  Eigen::Matrix<double, 3, 3> R_acc_ = Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> R_acc_inv_ =
      Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> R_ang_ = Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> R_ang_inv_ =
      Eigen::Matrix<double, 3, 3>::Identity();
  StaticNoiseModel<3>::Ptr acc_noise_model_ =
      StaticNoiseModel<3>::MakeShared(R_acc_);
  StaticNoiseModel<3>::Ptr gyro_noise_model_ =
      StaticNoiseModel<3>::MakeShared(R_ang_);

  getMotionPriorJacobians_(const lgmath::se3::Transformation &T1,
                           const lgmath::se3::Transformation &T2,
                           const Eigen::Matrix<double, 6, 1> &w2,
                           const Eigen::Matrix<double, 6, 1> &dw2,
                           const Eigen::Matrix<double, 18, 18> &Phi,
                           Eigen::Matrix<double, 18, 18> &F,
                           Eigen::Matrix<double, 18, 18> &E) const;
};

}  // namespace steam
