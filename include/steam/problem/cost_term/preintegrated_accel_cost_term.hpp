#pragma once

#include <steam/trajectory/const_vel/helper.hpp>
#include "steam/evaluable/se3/se3_state_var.hpp"
#include "steam/evaluable/state_var.hpp"
#include "steam/evaluable/vspace/vspace_state_var.hpp"
#include "steam/problem/cost_term/base_cost_term.hpp"
#include "steam/problem/cost_term/imu_super_cost_term.hpp"
#include "steam/problem/loss_func/loss_funcs.hpp"
#include "steam/problem/noise_model/static_noise_model.hpp"
#include "steam/problem/problem.hpp"
#include "steam/trajectory/const_vel/interface.hpp"
#include "steam/trajectory/time.hpp"

#include <iostream>

namespace steam {

class PreintAccCostTerm : public BaseCostTerm {
 public:
  enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options {
    int num_threads = 1;
    LOSS_FUNC loss_func = LOSS_FUNC::L2;
    double loss_sigma = 1.0;
    Eigen::Matrix<double, 3, 1> gravity = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
    bool se2 = false;
  };

  using Ptr = std::shared_ptr<PreintAccCostTerm>;
  using ConstPtr = std::shared_ptr<const PreintAccCostTerm>;

  using PoseType = lgmath::se3::Transformation;
  using VelType = Eigen::Matrix<double, 6, 1>;
  using BiasType = Eigen::Matrix<double, 6, 1>;

  using Interface = steam::traj::const_vel::Interface;

  using Variable = steam::traj::const_vel::Variable;

  using Time = steam::traj::Time;

  using Matrix12d = Eigen::Matrix<double, 12, 12>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  static Ptr MakeShared(const Interface::ConstPtr &interface, const Time &time1,
                        const Time &time2,
                        const Evaluable<BiasType>::ConstPtr &bias1,
                        const Evaluable<BiasType>::ConstPtr &bias2,
                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_1,
                        const Evaluable<PoseType>::ConstPtr &transform_i_to_m_2,
                        const Options &options);

  PreintAccCostTerm(const Interface::ConstPtr &interface, const Time &time1,
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
    Qinv_T_ = steam::traj::const_vel::getQinv(T, ones);
    Tran_T_ = steam::traj::const_vel::getTran(T);

    loss_func_ = [this]() -> BaseLossFunc::Ptr {
      switch (options_.loss_func) {
        case LOSS_FUNC::L2:
          return L2LossFunc::MakeShared();
        case LOSS_FUNC::DCS:
          return DcsLossFunc::MakeShared(options_.loss_sigma);
        case LOSS_FUNC::CAUCHY:
          return CauchyLossFunc::MakeShared(options_.loss_sigma);
        case LOSS_FUNC::GM:
          return GemanMcClureLossFunc::MakeShared(options_.loss_sigma);
        default:
          return nullptr;
      }
      return nullptr;
    }();

    jac_bias_accel_.block<3, 3>(0, 0) =
        Eigen::Matrix<double, 3, 3>::Identity() * -1;

    if (options_.se2) {
      jac_bias_accel_(2, 2) = 0.;
    }

    R_acc_.diagonal() = options_.r_imu_acc;
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
  Matrix12d Qinv_T_ = Matrix12d::Identity();
  Matrix12d Tran_T_ = Matrix12d::Identity();
  std::map<double, std::pair<Matrix12d, Matrix12d>> interp_mats_;

  std::vector<IMUData> imu_data_vec_;
  std::vector<double> meas_times_;

  BaseLossFunc::Ptr loss_func_ = L2LossFunc::MakeShared();
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  StaticNoiseModel<3>::Ptr acc_noise_model_ =
      StaticNoiseModel<3>::MakeShared(R);

  Eigen::Matrix<double, 3, 6> jac_bias_accel_ =
      Eigen::Matrix<double, 3, 6>::Zero();

  Eigen::Matrix3d R_acc_ = Eigen::Matrix3d::Zero();

  void initialize_interp_matrices_();
};

}  // namespace steam
