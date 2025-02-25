#include "steam/evaluable/p2p/gyro_error_evaluator.hpp"
#include <iostream>

namespace steam {
namespace p2p {

auto GyroErrorEvaluator::MakeShared(
    const Eigen::Vector3d &gyro_meas,
    const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
    const Evaluable<PoseInType>::ConstPtr &T_ms_curr) -> Ptr {
  return std::make_shared<GyroErrorEvaluator>(gyro_meas, T_ms_prev, T_ms_curr);
}

GyroErrorEvaluator::GyroErrorEvaluator(
    const Eigen::Vector3d &gyro_meas,
    const Evaluable<PoseInType>::ConstPtr &T_ms_prev,
    const Evaluable<PoseInType>::ConstPtr &T_ms_curr)
    : gyro_meas_(gyro_meas), T_ms_prev_(T_ms_prev), T_ms_curr_(T_ms_curr) {
}

bool GyroErrorEvaluator::active() const { return T_ms_prev_->active() || T_ms_curr_->active(); }

void GyroErrorEvaluator::getRelatedVarKeys(KeySet &keys) const {
    T_ms_prev_->getRelatedVarKeys(keys);
    T_ms_curr_->getRelatedVarKeys(keys);
}

auto GyroErrorEvaluator::value() const -> OutType {
    // Form measured and predicted printegrated DCM: prev (p) curr (c)
    Eigen::Vector3d meas_vec = gyro_meas_;
    const lgmath::so3::Rotation C_pc_meas(meas_vec);
    const lgmath::so3::Rotation C_cp_eval((T_ms_curr_->value().C_ba().inverse() * 
                                           T_ms_prev_->value().C_ba()).eval());
    std::cout << "Gyro error: " << -lgmath::so3::rot2vec((C_cp_eval * C_pc_meas).matrix()).transpose() << std::endl;

    // Return error (minus sign bc rot2vec flips rotation direction)
    return -lgmath::so3::rot2vec((C_cp_eval * C_pc_meas).matrix());
}

auto GyroErrorEvaluator::forward() const -> Node<OutType>::Ptr {
    const auto child1 = T_ms_prev_->forward();
    const auto child2 = T_ms_curr_->forward();

    const auto C_ms_prev = child1->value().C_ba();
    const auto C_ms_curr = child2->value().C_ba();

    // Form measured and predicted printegrated DCM: prev (p) curr (c)
    // clang-format off
    Eigen::Vector3d meas_vec = gyro_meas_;
    const lgmath::so3::Rotation C_pc_meas(meas_vec);
    const lgmath::so3::Rotation C_cp_eval((C_ms_curr.transpose() * C_ms_prev).eval());
    // Compute error (minus sign bc rot2vec flips rotation direction)
    OutType error = -lgmath::so3::rot2vec((C_cp_eval * C_pc_meas).matrix());
    // clang-format on

    const auto node = Node<OutType>::MakeShared(error);
    node->addChild(child1);
    node->addChild(child2);
    return node;
}

void GyroErrorEvaluator::backward(const Eigen::MatrixXd &lhs,
                                       const Node<OutType>::Ptr &node,
                                       Jacobians &jacs) const {
    if (T_ms_prev_->active()) {
        const auto child1 = std::static_pointer_cast<Node<PoseInType>>(node->at(0));
        
        Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();

        Eigen::Vector3d meas_vec = gyro_meas_;
        jac.block<3, 3>(0, 3) = -1 * lgmath::so3::vec2rot(meas_vec).transpose();

        T_ms_prev_->backward(lhs * jac, child1, jacs);
    }

    if (T_ms_curr_->active()) {
        const auto child2 = std::static_pointer_cast<Node<PoseInType>>(node->at(1));
        Eigen::Matrix<double, 3, 6> jac = Eigen::Matrix<double, 3, 6>::Zero();
        
        jac.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

        T_ms_curr_->backward(lhs * jac, child2, jacs);
    }
}

GyroErrorEvaluator::Ptr gyroError(
    const Eigen::Vector3d &gyro_meas,
    const Evaluable<GyroErrorEvaluator::PoseInType>::ConstPtr &T_ms_prev,
    const Evaluable<GyroErrorEvaluator::PoseInType>::ConstPtr &T_ms_curr) {
  return GyroErrorEvaluator::MakeShared(gyro_meas, T_ms_prev, T_ms_curr);
}

}  // namespace p2p
}  // namespace steam