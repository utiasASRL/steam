#pragma once

#include "steam/evaluable/state_var.hpp"

namespace steam {
namespace stereo {

class HomoPointStateVar : public StateVar<Eigen::Matrix<double, 4, 1>> {
 public:
  using Ptr = std::shared_ptr<HomoPointStateVar>;
  using ConstPtr = std::shared_ptr<const HomoPointStateVar>;

  using T = Eigen::Matrix<double, 4, 1>;
  using Base = StateVar<T>;

  static Ptr MakeShared(const Eigen::Vector3d& value, bool scale = false,
                        const std::string& name = "");
  /** \brief Constructor from a global 3D point */
  HomoPointStateVar(const Eigen::Vector3d& value, bool scale = false,
                    const std::string& name = "");

  /** \brief Update the landmark state from the 3-dimensional perturbation */
  bool update(const Eigen::VectorXd& perturbation) override;
  StateVarBase::Ptr clone() const override;

 private:
  /** \brief Refresh the homogeneous scaling */
  void refreshHomoScaling();

  bool scale_ = false;
};

}  // namespace stereo
}  // namespace steam
