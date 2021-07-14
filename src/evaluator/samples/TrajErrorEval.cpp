#include <steam/evaluator/samples/TrajErrorEval.hpp>


namespace steam {

TrajErrorEval::TrajErrorEval(const std::vector<se3::TransformEvaluator::Ptr>& poses,
                             const std::vector<VectorSpaceStateVar::Ptr>& vels) : vels_(vels) {

  assert(poses.size() == vels.size());  // todo - handle more cleanly
  n_ = poses.size();

  for (const auto & pose : poses) {
    se3::FixedTransformEvaluator::ConstPtr meas = se3::FixedTransformEvaluator::MakeShared(pose->evaluate());
    pose_evals_.push_back(se3::tran2vec(se3::composeInverse(meas, pose)));
  }
  for (const auto & vel : vels) {
    vel_evals_.push_back(steam::VectorSpaceErrorEval<6,6>::Ptr(new steam::VectorSpaceErrorEval<6,6>(vel->getValue(), vel)));
  }
}

bool TrajErrorEval::isActive() const {
  bool active = false;
  for (const auto & eval : pose_evals_) {   // todo: this can happen more efficiently
    active = active || eval->isActive();
  }
  for (const auto & eval : vel_evals_) {
    active = active || eval->isActive();
  }
  return active;
}

Eigen::Matrix<double, Eigen::Dynamic, 1> TrajErrorEval::evaluate() const {
  Eigen::Matrix<double, Eigen::Dynamic, 1> e = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(n_*12, 1);
  for (uint i = 0; i < n_; ++i) {
    e.block<6,1>(12*i, 0) = pose_evals_[i]->evaluate();
    e.block<6,1>(12*i + 6, 0) = vel_evals_[i]->evaluate();
  }

  return e;
}

Eigen::Matrix<double, Eigen::Dynamic, 1> TrajErrorEval::evaluate(
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &lhs,
    std::vector<Jacobian<Eigen::Dynamic, 6>> *jacs) const {
  // Check and initialize Jacobian array
  if (jacs == nullptr) {
    throw std::invalid_argument(
        "Null pointer provided to return-input 'jacs' in evaluate");
  }
  jacs->clear();

  for (uint i = 0; i < n_; ++i) {
    // If current pose unlocked, add Jacobian from perturbing it
    if (pose_evals_[i]->isActive()) {
      Eigen::Matrix<double, Eigen::Dynamic, 6> J = Eigen::Matrix<double, Eigen::Dynamic, 6>::Zero(12*n_,6);
      Eigen::Matrix<double, 6, 6> J_i = lgmath::se3::vec2jacinv(pose_evals_[i]->evaluateTree()->getValue());
      J.block<6,6>(12*i, 0) = J_i;
      jacs->resize(jacs->size() + 1);
      Jacobian<Eigen::Dynamic,6>& jacref = jacs->back();
      std::map<unsigned int, steam::StateVariableBase::Ptr> tmp_state;

      pose_evals_[i]->getActiveStateVariables(&tmp_state);
      jacref.key = tmp_state.begin()->second->getKey();
      jacref.jac = -1 * lhs * J;
    }
    if (vel_evals_[i]->isActive()) {
      Eigen::Matrix<double, Eigen::Dynamic, 6> J = Eigen::Matrix<double, Eigen::Dynamic, 6>::Zero(12*n_,6);
      J.block<6,6>(12*i + 6, 0) = Eigen::Matrix<double, 6, 6>::Identity();
      jacs->resize(jacs->size() + 1);
      Jacobian<Eigen::Dynamic,6>& jacref = jacs->back();
      jacref.key = vels_[i]->getKey();
      jacref.jac = -1 * lhs * J;
    }
  }

  return evaluate();
}

} // steam
