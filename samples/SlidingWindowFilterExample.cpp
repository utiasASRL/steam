/**
 * \file SlidingWindowFilterExample.cpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#include <iomanip>
#include <iostream>

#include "lgmath.hpp"
#include "steam.hpp"

using namespace steam;

// clang-format off
class MotionError : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<MotionError>;
  using ConstPtr = std::shared_ptr<const MotionError>;

  using InType = Eigen::Matrix<double, 1, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& xkm1,
                        const Evaluable<InType>::ConstPtr& xk, double vk) {
    return std::make_shared<MotionError>(xkm1, xk, vk);
  }
  MotionError(const Evaluable<InType>::ConstPtr& xkm1,
              const Evaluable<InType>::ConstPtr& xk, double vk)
      : xkm1_(xkm1), xk_(xk), vk_(vk) {}

  bool active() const override { return xkm1_->active() || xk_->active(); }
  void getRelatedVarKeys(KeySet& keys) const override {
    xkm1_->getRelatedVarKeys(keys);
    xk_->getRelatedVarKeys(keys);
  }

  OutType value() const override {
    double xkm1 = xkm1_->value().value();
    double xk = xk_->value().value();
    OutType v{xk - xkm1 - vk_};
    return v;
  }

  Node<OutType>::Ptr forward() const override {
    const auto xkm1_node = xkm1_->forward();
    const auto xk_node = xk_->forward();

    double xkm1 = xkm1_node->value().value();
    double xk = xk_node->value().value();
    OutType v{xk - xkm1 - vk_};

    const auto node = Node<OutType>::MakeShared(v);
    node->addChild(xkm1_node);
    node->addChild(xk_node);

    return node;
  }

  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const {
    if (xkm1_->active()) {
      const auto xkm1_node = std::static_pointer_cast<Node<InType>>(node->at(0));
      xkm1_->backward(-lhs, xkm1_node, jacs);
    }
    if (xk_->active()) {
      const auto xk_node = std::static_pointer_cast<Node<InType>>(node->at(0));
      xk_->backward(lhs, xk_node, jacs);
    }
  }

 private:
  const Evaluable<InType>::ConstPtr xkm1_;
  const Evaluable<InType>::ConstPtr xk_;
  const double vk_;
};
// clang-format on

class ObservationError : public Evaluable<Eigen::Matrix<double, 1, 1>> {
 public:
  using Ptr = std::shared_ptr<ObservationError>;
  using ConstPtr = std::shared_ptr<const ObservationError>;

  using InType = Eigen::Matrix<double, 1, 1>;
  using OutType = Eigen::Matrix<double, 1, 1>;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& xk, double yk) {
    return std::make_shared<ObservationError>(xk, yk);
  }
  ObservationError(const Evaluable<InType>::ConstPtr& xk, double yk)
      : xk_(xk), yk_(yk) {}

  bool active() const override { return xk_->active(); }
  void getRelatedVarKeys(KeySet& keys) const override {
    xk_->getRelatedVarKeys(keys);
  }

  OutType value() const override {
    double xk = xk_->value().value();
    OutType v{yk_ - xk};
    return v;
  }

  Node<OutType>::Ptr forward() const override {
    const auto xk_node = xk_->forward();

    double xk = xk_node->value().value();
    OutType v{yk_ - xk};

    const auto node = Node<OutType>::MakeShared(v);
    node->addChild(xk_node);

    return node;
  }

  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const {
    if (xk_->active()) {
      const auto xk_node = std::static_pointer_cast<Node<InType>>(node->at(0));
      xk_->backward(-lhs, xk_node, jacs);
    }
  }

 private:
  const Evaluable<InType>::ConstPtr xk_;
  const double yk_;
};

// clang-format off
int main(int argc, char** argv) {
  // initial state
  const double x0 = 1.1615000208;
  const double x0_cov = 0.001;
  // motion
  const std::vector<double> v{-0.0223729186, -0.019305205 , -0.0160666676, -0.0130467915, -0.0104000045,
                              -0.0074697884, -0.0044205347, -0.0018041659,  0.0007157883,  0.0039760454};
  const double v_cov = 0.0000226134;
  // measurement
  const std::vector<double> y{1.1439731207, 1.1239731207, 1.1079731207, 1.0959731207, 1.0769731207,
                              1.0679731207, 1.0669731207, 1.0609731207, 1.0619731207, 1.0679731207};
  const double r_cov = 0.0003669233;

  using State = Eigen::Matrix<double, 1, 1>;
  using Cov = Eigen::Matrix<double, 1, 1>;

  //
  SlidingWindowFilter swf(4);

  std::vector<vspace::VSpaceStateVar<1>::Ptr> x_var;
  x_var.emplace_back(vspace::VSpaceStateVar<1>::MakeShared(State{0.0}, "x0"));

  // add initial state variable
  swf.addVariable(x_var[0]);

  // add initial prior
  const auto loss_func = L2LossFunc::MakeShared();
  const auto noise_model = steam::StaticNoiseModel<1>::MakeShared(Cov{x0_cov});
  const auto error_func = vspace::vspace_error<1>(x_var[0], State{x0});
  const auto cost_term = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, loss_func);
  swf.addCostTerm(cost_term);

  for (int k = 1; k < (int)v.size(); ++k) {
    // add a new variable
    x_var.emplace_back(vspace::VSpaceStateVar<1>::MakeShared(State{0.0}, "x"+std::to_string(k)));
    swf.addVariable(x_var[k]);

    const auto xkm1_var = x_var[k - 1];
    const auto xk_var = x_var[k];

    // motion
    {
      const auto loss_func = L2LossFunc::MakeShared();
      const auto noise_model = steam::StaticNoiseModel<1>::MakeShared(Cov{v_cov});
      const auto error_func = MotionError::MakeShared(xkm1_var, xk_var, v[k]);
      const auto cost_term = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, loss_func);
      swf.addCostTerm(cost_term);
    }

    // observation
    {
      const auto loss_func = L2LossFunc::MakeShared();
      const auto noise_model = steam::StaticNoiseModel<1>::MakeShared(Cov{r_cov});
      const auto error_func = ObservationError::MakeShared(xk_var, y[k]);
      const auto cost_term = WeightedLeastSqCostTerm<1>::MakeShared(error_func, noise_model, loss_func);
      swf.addCostTerm(cost_term);
    }

    // marginalize out the previous variable
    swf.marginalizeVariable(xkm1_var);

    GaussNewtonSolver::Params params;
    params.max_iterations = 1;
    GaussNewtonSolver solver(swf, params);
    solver.optimize();

    //
    std::cout << "x" << k << " = " << xk_var->value().value() << std::endl;
  }

  return 0;
}