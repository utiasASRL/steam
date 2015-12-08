//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvaluators.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_TRANSFORM_EVALUATORS_HPP
#define STEAM_TRANSFORM_EVALUATORS_HPP

#include <Eigen/Core>

#include <steam/evaluator/BlockAutomaticEvaluator.hpp>
#include <steam/state/LieGroupStateVar.hpp>
#include <steam/state/VectorSpaceStateVar.hpp>
#include <steam/common/Time.hpp>

namespace steam {
namespace se3 {

// Basic types

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluator for a transformation matrix
///
/// *Note that we fix MAX_STATE_DIM to 6. Typically the performance benefits of fixed size
///  matrices begin to die if larger than 6x6. Size 6 allows for transformation matrices
///  and 6D velocities. If you have a state larger than this, consider writing an
///  error evaluator that extends from ErrorEvaluatorX.
//////////////////////////////////////////////////////////////////////////////////////////////
typedef BlockAutomaticEvaluator<lgmath::se3::Transformation, 6, 6> TransformEvaluator;

// Transformation evaluators

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Simple transform evaluator for a transformation state variable
//////////////////////////////////////////////////////////////////////////////////////////////
class TransformStateEvaluator : public TransformEvaluator
{
 public:

  /// Convenience typedefs
  typedef boost::shared_ptr<TransformStateEvaluator> Ptr;
  typedef boost::shared_ptr<const TransformStateEvaluator> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  TransformStateEvaluator(const se3::TransformStateVar::Ptr& transform);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pseudo constructor - return a shared pointer to a new instance
  //////////////////////////////////////////////////////////////////////////////////////////////
  static Ptr MakeShared(const se3::TransformStateVar::Ptr& transform);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Adds references (shared pointers) to active state variables to the map output
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void getActiveStateVariables(
      std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the transformation matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual lgmath::se3::Transformation evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the transformation matrix tree
  ///
  /// ** Note that the returned pointer belongs to the memory pool EvalTreeNode<TYPE>::pool,
  ///    and should be given back to the pool, rather than being deleted.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual EvalTreeNode<lgmath::se3::Transformation>* evaluateTree() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,1,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<1,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,2,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<2,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,3,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<3,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,4,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<4,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,6,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<6,6> >* outJacobians) const;

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Implementation for Block Automatic Differentiation
  //////////////////////////////////////////////////////////////////////////////////////////////
  template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
  void appendJacobiansImpl(const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
                           EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                           std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Transformation matrix state variable
  //////////////////////////////////////////////////////////////////////////////////////////////
  se3::TransformStateVar::Ptr transform_;

};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Simple transform evaluator for a fixed transformation
//////////////////////////////////////////////////////////////////////////////////////////////
class FixedTransformEvaluator : public TransformEvaluator
{
 public:

  /// Convenience typedefs
  typedef boost::shared_ptr<FixedTransformEvaluator> Ptr;
  typedef boost::shared_ptr<const FixedTransformEvaluator> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  FixedTransformEvaluator(const lgmath::se3::Transformation& transform);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pseudo constructor - return a shared pointer to a new instance
  //////////////////////////////////////////////////////////////////////////////////////////////
  static Ptr MakeShared(const lgmath::se3::Transformation& transform);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Adds references (shared pointers) to active state variables to the map output
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void getActiveStateVariables(
      std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the transformation matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual lgmath::se3::Transformation evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the transformation matrix tree
  ///
  /// ** Note that the returned pointer belongs to the memory pool EvalTreeNode<TYPE>::pool,
  ///    and should be given back to the pool, rather than being deleted.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual EvalTreeNode<lgmath::se3::Transformation>* evaluateTree() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,1,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<1,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,2,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<2,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,3,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<3,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,4,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<4,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,6,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<6,6> >* outJacobians) const;

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Implementation for Block Automatic Differentiation
  //////////////////////////////////////////////////////////////////////////////////////////////
  template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
  void appendJacobiansImpl(const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
                           EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                           std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Fixed transformation matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  lgmath::se3::Transformation transform_;
};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Simple transform evaluator for a transformation state variable
//////////////////////////////////////////////////////////////////////////////////////////////
class ConstVelTransformEvaluator : public TransformEvaluator
{
 public:

  /// Convenience typedefs
  typedef boost::shared_ptr<ConstVelTransformEvaluator> Ptr;
  typedef boost::shared_ptr<const ConstVelTransformEvaluator> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  ConstVelTransformEvaluator(const VectorSpaceStateVar::Ptr& velocity, const Time& time);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pseudo constructor - return a shared pointer to a new instance
  //////////////////////////////////////////////////////////////////////////////////////////////
  static Ptr MakeShared(const VectorSpaceStateVar::Ptr& velocity, const Time& time);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Adds references (shared pointers) to active state variables to the map output
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void getActiveStateVariables(
      std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the transformation matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual lgmath::se3::Transformation evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the transformation matrix tree
  ///
  /// ** Note that the returned pointer belongs to the memory pool EvalTreeNode<TYPE>::pool,
  ///    and should be given back to the pool, rather than being deleted.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual EvalTreeNode<lgmath::se3::Transformation>* evaluateTree() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,1,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<1,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,2,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<2,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,3,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<3,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,4,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<4,6> >* outJacobians) const;

  virtual void appendJacobians(const Eigen::Matrix<double,6,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<6,6> >* outJacobians) const;

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Implementation for Block Automatic Differentiation
  //////////////////////////////////////////////////////////////////////////////////////////////
  template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
  void appendJacobiansImpl(const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
                           EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                           std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Velocity state variable
  //////////////////////////////////////////////////////////////////////////////////////////////
  VectorSpaceStateVar::Ptr velocity_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Duration
  //////////////////////////////////////////////////////////////////////////////////////////////
  Time time_;
};

} // se3
} // steam

#endif // STEAM_TRANSFORM_EVALUATORS_HPP
