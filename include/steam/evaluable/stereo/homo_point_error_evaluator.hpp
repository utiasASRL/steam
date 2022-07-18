#pragma once

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace stereo {

class HomoPointErrorEvaluator : public Evaluable<Eigen::Vector3d> {
 public:
  using Ptr = std::shared_ptr<HomoPointErrorEvaluator>;
  using ConstPtr = std::shared_ptr<const HomoPointErrorEvaluator>;

  using InType = Eigen::Vector4d;
  using OutType = Eigen::Vector3d;

  static Ptr MakeShared(const Evaluable<InType>::ConstPtr& pt,
                        const InType& meas_pt);
  HomoPointErrorEvaluator(const Evaluable<InType>::ConstPtr& pt,
                          const InType& meas_pt);

  bool active() const override;
  void getRelatedVarKeys(KeySet& keys) const override;

  OutType value() const override;
  Node<OutType>::Ptr forward() const override;
  void backward(const Eigen::MatrixXd& lhs, const Node<OutType>::Ptr& node,
                Jacobians& jacs) const override;

 private:
  /** \brief Transform evaluable */
  const Evaluable<InType>::ConstPtr pt_;
  /** \brief Landmark state variable */
  const InType meas_pt_;
  // constants
  Eigen::Matrix<double, 3, 4> D_ = Eigen::Matrix<double, 3, 4>::Zero();
};

HomoPointErrorEvaluator::Ptr homo_point_error(
    const Evaluable<HomoPointErrorEvaluator::InType>::ConstPtr& pt,
    const HomoPointErrorEvaluator::InType& meas_pt);

}  // namespace stereo
}  // namespace steam
