#pragma once

#include "steam/evaluable/se3/se3_state_var.hpp"
#include "steam/evaluable/state_var.hpp"
#include "steam/evaluable/vspace/vspace_state_var.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/noise_model/static_noise_model.hpp"
#include "steam/problem/problem.hpp"
#include "steam/trajectory/const_acc/interface.hpp"
#include "steam/trajectory/time.hpp"

#include <iostream>

namespace steam {

struct IMUData {
  double timestamp = 0;
  Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d lin_acc = Eigen::Vector3d::Zero();

  IMUData(double timestamp_, Eigen::Vector3d ang_vel_, Eigen::Vector3d lin_acc_)
      : timestamp(timestamp_), ang_vel(ang_vel_), lin_acc(lin_acc_) {}

  IMUData() {}
};

class IMUSuperCostTerm : public BaseCostTerm {
 public:
  enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options {
    int num_threads = 1;
    LOSS_FUNC acc_loss_func = LOSS_FUNC::CAUCHY;
    LOSS_FUNC gyro_loss_func = LOSS_FUNC::CAUCHY;
    double acc_loss_sigma = 0.1;
    double gyro_loss_sigma = 0.1;
    Eigen::Matrix<double, 3, 1> gravity = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
    bool se2 = false;
  };

  using Ptr = std::shared_ptr<IMUSuperCostTerm>;
  using ConstPtr = std::shared_ptr<const IMUSuperCostTerm>;

  using PoseType = lgmath::se3::Transformation;
  using VelType = Eigen::Matrix<double, 6, 1>;
  using AccType = Eigen::Matrix<double, 6, 1>;
  using BiasType = Eigen::Matrix<double, 6, 1>;

  using Interface = steam::traj::const_acc::Interface;

  using Variable = steam::traj::const_acc::Variable;

  using Time = steam::traj::Time;

  using Matrix18d = Eigen::Matrix<double, 18, 18>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  static Ptr MakeShared(const Interface::ConstPtr &interface, const Time &time1,
                        const Time &time2,
                        const Evaluable<BiasType>::ConstPtr &bias1,
                        const Evaluable<BiasType>::ConstPtr &bias2,
                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
                        const Options &options);

  IMUSuperCostTerm(const Interface::ConstPtr &interface, const Time &time1,
                   const Time &time2,
                   const Evaluable<BiasType>::ConstPtr &bias1,
                   const Evaluable<BiasType>::ConstPtr &bias2,
                   const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
                   const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
                   const Options &options)
      : interface_(interface),
        time1_(time1),
        time2_(time2),
        bias1_(bias1),
        bias2_(bias2),
        transform_i_to_m_1_(transform_i_to_m_1),
        transform_i_to_m_2_(transform_i_to_m_2),
        options_(options),
        knot1_(interface_->get(time1)),
        knot2_(interface_->get(time2)) {
    const double T = (knot2_->time() - knot1_->time()).seconds();
    const Eigen::Matrix<double, 6, 1> ones =
        Eigen::Matrix<double, 6, 1>::Ones();
    Qinv_T_ = interface_->getQinvPublic(T, ones);
    Tran_T_ = interface_->getTranPublic(T);

    acc_loss_func_ = [this]() -> BaseLossFunc::Ptr {
      switch (options_.acc_loss_func) {
        case LOSS_FUNC::L2:
          return L2LossFunc::MakeShared();
        case LOSS_FUNC::DCS:
          return DcsLossFunc::MakeShared(options_.acc_loss_sigma);
        case LOSS_FUNC::CAUCHY:
          return CauchyLossFunc::MakeShared(options_.acc_loss_sigma);
        case LOSS_FUNC::GM:
          return GemanMcClureLossFunc::MakeShared(options_.acc_loss_sigma);
        default:
          return nullptr;
      }
      return nullptr;
    }();

    gyro_loss_func_ = [this]() -> BaseLossFunc::Ptr {
      switch (options_.gyro_loss_func) {
        case LOSS_FUNC::L2:
          return L2LossFunc::MakeShared();
        case LOSS_FUNC::DCS:
          return DcsLossFunc::MakeShared(options_.gyro_loss_sigma);
        case LOSS_FUNC::CAUCHY:
          return CauchyLossFunc::MakeShared(options_.gyro_loss_sigma);
        case LOSS_FUNC::GM:
          return GemanMcClureLossFunc::MakeShared(options_.gyro_loss_sigma);
        default:
          return nullptr;
      }
      return nullptr;
    }();

    jac_vel_.block<3, 3>(0, 3) = Eigen::Matrix<double, 3, 3>::Identity();
    jac_accel_.block<3, 3>(0, 0) = Eigen::Matrix<double, 3, 3>::Identity();
    jac_bias_accel_.block<3, 3>(0, 0) =
        Eigen::Matrix<double, 3, 3>::Identity() * -1;
    jac_bias_gyro_.block<3, 3>(0, 3) =
        Eigen::Matrix<double, 3, 3>::Identity() * -1;

    Eigen::Matrix3d R_acc = Eigen::Matrix3d::Zero();
    R_acc.diagonal() = options_.r_imu_acc;
    Eigen::Matrix3d R_gyro = Eigen::Matrix3d::Zero();
    R_gyro.diagonal() = options_.r_imu_ang;
    acc_noise_model_ = StaticNoiseModel<3>::MakeShared(R_acc);
    gyro_noise_model_ = StaticNoiseModel<3>::MakeShared(R_gyro);
  }

  /** \brief Compute the cost to the objective function */
  double cost() const override;

  /** \brief Get keys of variables related to this cost term */
  void getRelatedVarKeys(KeySet &keys) const override;

  void init();

  void emplace_back(IMUData &imu_data) { imu_data_vec_.emplace_back(imu_data); }

  void clear() { imu_data_vec_.clear(); }
  void reserve(unsigned int N) { imu_data_vec_.reserve(N); }

  std::vector<IMUData> &get() { return imu_data_vec_; }
  void set(const std::vector<IMUData> imu_data_vec) {
    imu_data_vec_ = imu_data_vec;
  }

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
  const Evaluable<BiasType>::ConstPtr bias1_;
  const Evaluable<BiasType>::ConstPtr bias2_;
  const Evaluable<PoseType>::ConstPtr transform_i_to_m_1_;
  const Evaluable<PoseType>::ConstPtr transform_i_to_m_2_;
  const Options options_;
  const Variable::ConstPtr knot1_;
  const Variable::ConstPtr knot2_;
  Matrix18d Qinv_T_ = Matrix18d::Identity();
  Matrix18d Tran_T_ = Matrix18d::Identity();
  std::map<double, std::pair<Matrix18d, Matrix18d>> interp_mats_;

  std::vector<IMUData> imu_data_vec_;
  std::vector<double> meas_times_;

  BaseLossFunc::Ptr acc_loss_func_ = L2LossFunc::MakeShared();
  BaseLossFunc::Ptr gyro_loss_func_ = L2LossFunc::MakeShared();
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  StaticNoiseModel<3>::Ptr acc_noise_model_ =
      StaticNoiseModel<3>::MakeShared(R);
  StaticNoiseModel<3>::Ptr gyro_noise_model_ =
      StaticNoiseModel<3>::MakeShared(R);

  Eigen::Matrix<double, 3, 6> jac_vel_ = Eigen::Matrix<double, 3, 6>::Zero();
  Eigen::Matrix<double, 3, 6> jac_accel_ = Eigen::Matrix<double, 3, 6>::Zero();
  Eigen::Matrix<double, 3, 6> jac_bias_accel_ =
      Eigen::Matrix<double, 3, 6>::Zero();
  Eigen::Matrix<double, 3, 6> jac_bias_gyro_ =
      Eigen::Matrix<double, 3, 6>::Zero();

  void initialize_interp_matrices_();
};

}  // namespace steam
