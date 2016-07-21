//////////////////////////////////////////////////////////////////////////////////////////////
/// \file PointToPointErrorEval.hpp
///
/// \author Francois Pomerleau, ASRL
/// \brief This evaluator was develop in the context of ICP (Iterative Closest Point) 
///        implementation.
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_POINT_TO_POINT_ERROR_EVALUATOR_HPP
#define STEAM_POINT_TO_POINT_ERROR_EVALUATOR_HPP

#include <steam.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief The distance between two points living in their respective frame is used as our
///        error function.
///
//////////////////////////////////////////////////////////////////////////////////////////////
class PointToPointErrorEval : public ErrorEvaluator<4,6>::type
{
public:

  /// Convenience typedefs
  typedef boost::shared_ptr<PointToPointErrorEval> Ptr;
  typedef boost::shared_ptr<const PointToPointErrorEval> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
	/// \param ref_a     A point from the reference point cloud (static) expressed in homogeneous 
	///                  coordinates (i.e., [x, y, z, 1]) and in the frame a.
	/// \param T_a_world Transformation matrix from frame world to frame a.
	/// \param read_b    A point from the reading point cloud expressed in homogeneous 
	///                  coordinates (i.e., [x, y, z, 1]) and in the frame b.
	/// \param T_b_world Transformation matrix from frame world to frame b.
  //////////////////////////////////////////////////////////////////////////////////////////////
  PointToPointErrorEval(const Eigen::Vector4d& ref_a,
                        const se3::TransformEvaluator::ConstPtr& T_a_world,
												const Eigen::Vector4d& read_b,
                        const se3::TransformEvaluator::ConstPtr& T_b_world
												);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const; //TODO: check if we need to define that

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 4-d measurement error (x, y, z, 0)
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Vector4d evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the 4-d measurement error (x, y, z, 0) and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Vector4d evaluate(const Eigen::Matrix4d& lhs,
                                   std::vector<Jacobian<4,6> >* jacs) const;

private:
  
  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Point evaluator (evaluates the point transformed into the camera frame)
  //////////////////////////////////////////////////////////////////////////////////////////////
	Eigen::Vector4d ref_a_;
  se3::ComposeInverseTransformEvaluator::ConstPtr T_ab_;
	Eigen::Vector4d read_b_;

};

} // steam

#endif // STEAM_POINT_TO_POINT_ERROR_EVALUATOR_HPP
