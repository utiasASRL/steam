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
#include "steam/trajectory/time.hpp"

#include <iostream>

namespace steam {

struct PreintegratedMeasurement {
  Eigen::Matrix3d C_ij;
  Eigen::Vector3d r_ij;
  Eigen::Vector3d v_ij;
  Eigen::Matrix<double, 9, 9> cov;

  PreintegratedMeasurement(Eigen::Matrix3d C_ij_, Eigen::Vector3d r_ij_, Eigen::Vector3d v_ij_, Eigen::Matrix<double, 9, 9> cov_) : C_ij(C_ij_), r_ij(r_ij_), v_ij(v_ij_), cov(cov_) {}
  PreintegratedMeasurement() {}
};

class PreintIMUCostTerm : public BaseCostTerm {
 public:
  enum class LOSS_FUNC { L2, DCS, CAUCHY, GM };

  struct Options {
    LOSS_FUNC loss_func = LOSS_FUNC::L2;
    double loss_sigma = 1.0;
    Eigen::Matrix<double, 3, 1> gravity = {0, 0, -9.8042};
    Eigen::Matrix<double, 3, 1> r_imu_acc = Eigen::Matrix<double, 3, 1>::Zero();
    Eigen::Matrix<double, 3, 1> r_imu_ang = Eigen::Matrix<double, 3, 1>::Zero();
  };

  using Ptr = std::shared_ptr<PreintIMUCostTerm>;
  using ConstPtr = std::shared_ptr<const PreintIMUCostTerm>;

  using PoseType = lgmath::se3::Transformation;
  using VelType = Eigen::Matrix<double, 3, 1>;
  using BiasType = Eigen::Matrix<double, 6, 1>;

  using Time = steam::traj::Time;


  static Ptr MakeShared(const Time time1,
                        const Time time2,
                        const Evaluable<PoseType>::ConstPtr &transform_r_to_m_1,
                        const Evaluable<PoseType>::ConstPtr &transform_r_to_m_2,
                        const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m_1,
                        const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m_2,
                        const Evaluable<BiasType>::ConstPtr &bias,
                        const Options &options);

  PreintIMUCostTerm(const Time time1,
                    const Time time2,
                    const Evaluable<PoseType>::ConstPtr &transform_r_to_m_1,
                    const Evaluable<PoseType>::ConstPtr &transform_r_to_m_2,
                    const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m_1,
                    const Evaluable<VelType>::ConstPtr &v_m_to_r_in_m_2,
                    const Evaluable<BiasType>::ConstPtr &bias,
                    const Options &options)
      : time1_(time1),
        time2_(time2),
        transform_r_to_m_1_(transform_r_to_m_1),
        transform_r_to_m_2_(transform_r_to_m_2),
        v_m_to_r_in_m_1_(v_m_to_r_in_m_1),
        v_m_to_r_in_m_2_(v_m_to_r_in_m_2),
        bias_(bias),
        options_(options) {

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

    R_acc_.diagonal() = options_.r_imu_acc;
    R_ang_.diagonal() = options_.r_imu_ang;
    gravity_ = options_.gravity;
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

  

  /** \brief Returns the 9x24 Jacbobian of the error with respect to the state variables
   * e = [e(r); e(p); e(v)], x = [dp_i; dr_i; dv_i; dp_j; dr_j; dv_j; db_a; db_g].
   * Returns: d e / d x
  */
  Eigen::Matrix<double, 9, 24> get_jacobian() const;

  /** \brief Returns the 9x1 error vector associated with this preintegrated IMU factor. 
   * e = [e(r); e(p); e(v)]
  */
  Eigen::Matrix<double, 9, 1> get_error() const;

  /** \brief Produces the preintegrated IMU measurement.
   * Returns a tuple of C_ij, r_ij, v_ij, covariance(dr, dp, dv).
   */
  PreintegratedMeasurement preintegrate_() const;


 private:
  const Time time1_;
  const Time time2_;
  const Evaluable<PoseType>::ConstPtr transform_r_to_m_1_;
  const Evaluable<PoseType>::ConstPtr transform_r_to_m_2_;
  const Evaluable<VelType>::ConstPtr v_m_to_r_in_m_1_;
  const Evaluable<VelType>::ConstPtr v_m_to_r_in_m_2_;
  const Evaluable<BiasType>::ConstPtr bias_;
  const Options options_;

  std::vector<IMUData> imu_data_vec_;

  BaseLossFunc::Ptr loss_func_ = L2LossFunc::MakeShared();

  Eigen::Matrix3d R_acc_ = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d R_ang_ = Eigen::Matrix3d::Zero();
  Eigen::Vector3d gravity_ = {0, 0, -9.8042};
  
};

}  // namespace steam
