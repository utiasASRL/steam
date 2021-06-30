//////////////////////////////////////////////////////////////////////////////////////////////
/// \file PointToPointErrorEval2.cpp
///
/// \author Yuchen Wu, ASRL
/// \brief This evaluator was developed in the context of ICP (Iterative
///        Closest Point) implementation. It is different from
///        PointToPointErrorEval in that it takes points in cartesian
///        coordinates instead of homogeneous coordinates.
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_POINT_TO_POINT_ERROR_EVALUATOR_2_HPP
#define STEAM_POINT_TO_POINT_ERROR_EVALUATOR_2_HPP

#include <steam.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The distance between two points living in their respective frame is
///        used as our error function.
//////////////////////////////////////////////////////////////////////////////////////////////
class PointToPointErrorEval2 : public ErrorEvaluator<3, 6>::type {
 public:
  /// Convenience typedefs
  typedef boost::shared_ptr<PointToPointErrorEval2> Ptr;
  typedef boost::shared_ptr<const PointToPointErrorEval2> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  /// \param T_rq      Transformation matrix from query to reference.
  /// \param reference A point in the 'reference' frame expressed in cartesian
  ///                  coordinates.
  /// \param query     A point in the 'query' frame expressed in cartesian
  ///                  coordinates.
  //////////////////////////////////////////////////////////////////////////////////////////////
  PointToPointErrorEval2(const se3::TransformEvaluator::ConstPtr &T_rq,
                         const Eigen::Vector3d &reference,
                         const Eigen::Vector3d &query);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state
  ///        variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  bool isActive() const override;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 3-d measurement error (x, y, z)
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 3, 1> evaluate() const override;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 3-d measurement error (x, y, z) and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Matrix<double, 3, 1> evaluate(
      const Eigen::Matrix<double, 3, 3> &lhs,
      std::vector<Jacobian<3, 6>> *jacs) const;

 private:
  se3::TransformEvaluator::ConstPtr T_rq_;

  Eigen::Matrix<double, 3, 4> D_ = Eigen::Matrix<double, 3, 4>::Zero();

  Eigen::Vector4d reference_ = Eigen::Vector4d::Constant(1);
  Eigen::Vector4d query_ = Eigen::Vector4d::Constant(1);
};

}  // namespace steam

#endif  // STEAM_POINT_TO_POINT_ERROR_EVALUATOR_2_HPP