//////////////////////////////////////////////////////////////////////////////////////////////
/// \file TransformEvalOperations.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_TRANSFORM_EVALUATOR_OPERATIONS_HPP
#define STEAM_TRANSFORM_EVALUATOR_OPERATIONS_HPP

#include <Eigen/Core>

#include <steam/evaluator/blockauto/transform/TransformEvaluator.hpp>
#include <steam/state/LandmarkStateVar.hpp>

namespace steam {
namespace se3 {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluator for the composition of transformation matrices
//////////////////////////////////////////////////////////////////////////////////////////////
class ComposeTransformEvaluator : public TransformEvaluator
{
public:

  /// Convenience typedefs
  typedef boost::shared_ptr<ComposeTransformEvaluator> Ptr;
  typedef boost::shared_ptr<const ComposeTransformEvaluator> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  ComposeTransformEvaluator(const TransformEvaluator::ConstPtr& transform1, const TransformEvaluator::ConstPtr& transform2);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pseudo constructor - return a shared pointer to a new instance
  //////////////////////////////////////////////////////////////////////////////////////////////
  static Ptr MakeShared(const TransformEvaluator::ConstPtr& transform1, const TransformEvaluator::ConstPtr& transform2);

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
  /// \brief Evaluate the resultant transformation matrix (transform1*transform2)
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
  virtual void appendBlockAutomaticJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,1,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<1,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,2,6>& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<2,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,3,6>& lhs,
                                EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                std::vector<Jacobian<3,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,4,6>& lhs,
                                EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                std::vector<Jacobian<4,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,6,6>& lhs,
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
  /// \brief First transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  TransformEvaluator::ConstPtr transform1_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Second transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  TransformEvaluator::ConstPtr transform2_;

};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluator for the inverse of a transformation matrix
//////////////////////////////////////////////////////////////////////////////////////////////
class InverseTransformEvaluator : public TransformEvaluator
{
public:

  /// Convenience typedefs
  typedef boost::shared_ptr<InverseTransformEvaluator> Ptr;
  typedef boost::shared_ptr<const InverseTransformEvaluator> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  InverseTransformEvaluator(const TransformEvaluator::ConstPtr& transform);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pseudo constructor - return a shared pointer to a new instance
  //////////////////////////////////////////////////////////////////////////////////////////////
  static Ptr MakeShared(const TransformEvaluator::ConstPtr& transform);

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
  /// \brief Evaluate the resultant transformation matrix
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
  virtual void appendBlockAutomaticJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Fixed-size evaluations of the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,1,6>& lhs,
                                EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                std::vector<Jacobian<1,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,2,6>& lhs,
                                EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                std::vector<Jacobian<2,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,3,6>& lhs,
                                EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                std::vector<Jacobian<3,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,4,6>& lhs,
                                EvalTreeNode<lgmath::se3::Transformation>* evaluationTree,
                                std::vector<Jacobian<4,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,6,6>& lhs,
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
  /// \brief Transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  TransformEvaluator::ConstPtr transform_;

};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluator for the logarithmic map of a transformation matrix
///
/// *Note that we fix MAX_STATE_DIM to 6. Typically the performance benefits of fixed size
///  matrices begin to die if larger than 6x6. Size 6 allows for transformation matrices
///  and 6D velocities. If you have a state larger than this, consider writing an
///  error evaluator that extends from ErrorEvaluatorX.
//////////////////////////////////////////////////////////////////////////////////////////////
class LogMapEvaluator : public BlockAutomaticEvaluator<Eigen::Matrix<double,6,1>, 6, 6>
{
public:

  /// Convenience typedefs
  typedef boost::shared_ptr<LogMapEvaluator> Ptr;
  typedef boost::shared_ptr<const LogMapEvaluator> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  LogMapEvaluator(const TransformEvaluator::ConstPtr& transform);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pseudo constructor - return a shared pointer to a new instance
  //////////////////////////////////////////////////////////////////////////////////////////////
  static Ptr MakeShared(const TransformEvaluator::ConstPtr& transform);

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
  /// \brief Evaluate the resultant 6x1 vector belonging to the se(3) algebra
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Matrix<double,6,1> evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the resultant 6x1 vector belonging to the se(3) algebra and
  ///        sub-tree of evaluations
  ///
  /// ** Note that the returned pointer belongs to the memory pool EvalTreeNode<TYPE>::pool,
  ///    and should be given back to the pool, rather than being deleted.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluateTree() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendBlockAutomaticJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Fixed-size evaluations of the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,1,6>& lhs,
                               EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                               std::vector<Jacobian<1,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,2,6>& lhs,
                               EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                               std::vector<Jacobian<2,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,3,6>& lhs,
                               EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                               std::vector<Jacobian<3,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,4,6>& lhs,
                               EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                               std::vector<Jacobian<4,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,6,6>& lhs,
                               EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                               std::vector<Jacobian<6,6> >* outJacobians) const;

private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Implementation for Block Automatic Differentiation
  //////////////////////////////////////////////////////////////////////////////////////////////
  template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
  void appendJacobiansImpl(const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
                           EvalTreeNode<Eigen::Matrix<double,6,1> >* evaluationTree,
                           std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  TransformEvaluator::ConstPtr transform_;

};

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Evaluator for the composition of a transformation evaluator and landmark state
///
/// *Note that we fix MAX_STATE_DIM to 6. Typically the performance benefits of fixed size
///  matrices begin to die if larger than 6x6. Size 6 allows for transformation matrices
///  and 6D velocities. If you have a state larger than this, consider writing an
///  error evaluator that extends from ErrorEvaluatorX.
//////////////////////////////////////////////////////////////////////////////////////////////
class ComposeLandmarkEvaluator : public BlockAutomaticEvaluator<Eigen::Vector4d, 4, 6>
{
public:

  /// Convenience typedefs
  typedef boost::shared_ptr<ComposeLandmarkEvaluator> Ptr;
  typedef boost::shared_ptr<const ComposeLandmarkEvaluator> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  ComposeLandmarkEvaluator(const TransformEvaluator::ConstPtr& transform, const se3::LandmarkStateVar::Ptr& landmark);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Pseudo constructor - return a shared pointer to a new instance
  //////////////////////////////////////////////////////////////////////////////////////////////
  static Ptr MakeShared(const TransformEvaluator::ConstPtr& transform, const se3::LandmarkStateVar::Ptr& landmark);

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
  /// \brief Evaluate the point transformed by the transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Eigen::Vector4d evaluate() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the point transformed by the transform evaluator and
  ///        sub-tree of evaluations
  ///
  /// ** Note that the returned pointer belongs to the memory pool EvalTreeNode<TYPE>::pool,
  ///    and should be given back to the pool, rather than being deleted.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual EvalTreeNode<Eigen::Vector4d>* evaluateTree() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Evaluate the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendBlockAutomaticJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Fixed-size evaluations of the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,1,4>& lhs,
                                EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                                std::vector<Jacobian<1,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,2,4>& lhs,
                                EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                                std::vector<Jacobian<2,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,3,4>& lhs,
                                EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                                std::vector<Jacobian<3,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,4,4>& lhs,
                                EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                                std::vector<Jacobian<4,6> >* outJacobians) const;

  virtual void appendBlockAutomaticJacobians(const Eigen::Matrix<double,6,4>& lhs,
                                EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                                std::vector<Jacobian<6,6> >* outJacobians) const;

private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Implementation for Block Automatic Differentiation
  //////////////////////////////////////////////////////////////////////////////////////////////
  template<int LHS_DIM, int INNER_DIM, int MAX_STATE_SIZE>
  void appendJacobiansImpl(const Eigen::Matrix<double,LHS_DIM,INNER_DIM>& lhs,
                           EvalTreeNode<Eigen::Vector4d>* evaluationTree,
                           std::vector<Jacobian<LHS_DIM,MAX_STATE_SIZE> >* outJacobians) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Transform evaluator
  //////////////////////////////////////////////////////////////////////////////////////////////
  TransformEvaluator::ConstPtr transform_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Landmark state variable
  //////////////////////////////////////////////////////////////////////////////////////////////
  se3::LandmarkStateVar::Ptr landmark_;

};


/// Quick Ops

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compose two transform evaluators
//////////////////////////////////////////////////////////////////////////////////////////////
static TransformEvaluator::Ptr compose(const TransformEvaluator::ConstPtr& transform1,
                                       const TransformEvaluator::ConstPtr& transform2) {
  return ComposeTransformEvaluator::MakeShared(transform1, transform2);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Compose a transform evaluator and landmark state variable
//////////////////////////////////////////////////////////////////////////////////////////////
static ComposeLandmarkEvaluator::Ptr compose(const TransformEvaluator::ConstPtr& transform,
                                             const se3::LandmarkStateVar::Ptr& landmark) {
  return ComposeLandmarkEvaluator::MakeShared(transform, landmark);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Invert a transform evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
static TransformEvaluator::Ptr inverse(const TransformEvaluator::ConstPtr& transform) {
  return InverseTransformEvaluator::MakeShared(transform);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Take the 'logarithmic map' of a transformation evaluator
//////////////////////////////////////////////////////////////////////////////////////////////
static LogMapEvaluator::Ptr tran2vec(const TransformEvaluator::ConstPtr& transform) {
  return LogMapEvaluator::MakeShared(transform);
}

} // se3
} // steam

#endif // STEAM_TRANSFORM_EVALUATOR_OPERATIONS_HPP
