//////////////////////////////////////////////////////////////////////////////////////////////
/// \file BlockAutomaticEvaluator.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_BLOCK_AUTOMATIC_EVALUATOR_HPP
#define STEAM_BLOCK_AUTOMATIC_EVALUATOR_HPP

#include <steam/evaluator/EvaluatorBase.hpp>
#include <steam/evaluator/jacobian/EvalTreeNode.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Base class that defines the general 'evaluator' interface
//////////////////////////////////////////////////////////////////////////////////////////////
template <typename TYPE, int INNER_DIM, int MAX_STATE_SIZE>
class BlockAutomaticEvaluator : public EvaluatorBase<TYPE>
{
 public:

  /// Convenience typedefs
  typedef boost::shared_ptr<BlockAutomaticEvaluator<TYPE,INNER_DIM,MAX_STATE_SIZE> > Ptr;
  typedef boost::shared_ptr<const BlockAutomaticEvaluator<TYPE,INNER_DIM,MAX_STATE_SIZE> > ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  BlockAutomaticEvaluator();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Returns whether or not an evaluator contains unlocked state variables
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isActive() const = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Adds references (shared pointers) to active state variables to the map output
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void getActiveStateVariables(
      std::map<unsigned int, steam::StateVariableBase::Ptr>* outStates) const = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Interface for the general 'evaluation'
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual TYPE evaluate() const = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief General evaluation and Jacobians
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual TYPE evaluate(const Eigen::MatrixXd& lhs,
                        std::vector<Jacobian<> >* jacs) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Interface for an evaluation method that returns the full tree of evaluations
  ///
  /// ** Note that the returned pointer belongs to the memory pool EvalTreeNode<TYPE>::pool,
  ///    and should be given back to the pool, rather than being deleted.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual EvalTreeNode<TYPE>* evaluateTree() const = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Interface for the evaluation of the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendJacobians(const Eigen::MatrixXd& lhs,
                               EvalTreeNode<TYPE>* evaluationTree,
                               std::vector<Jacobian<> >* outJacobians) const = 0;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Fixed-size interface for the evaluation of the Jacobian tree
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void appendJacobians(const Eigen::Matrix<double,1,INNER_DIM>& lhs,
                               EvalTreeNode<TYPE>* evaluationTree,
                               std::vector<Jacobian<1,MAX_STATE_SIZE> >* outJacobians) const = 0;
  virtual void appendJacobians(const Eigen::Matrix<double,2,INNER_DIM>& lhs,
                               EvalTreeNode<TYPE>* evaluationTree,
                               std::vector<Jacobian<2,MAX_STATE_SIZE> >* outJacobians) const = 0;
  virtual void appendJacobians(const Eigen::Matrix<double,3,INNER_DIM>& lhs,
                               EvalTreeNode<TYPE>* evaluationTree,
                               std::vector<Jacobian<3,MAX_STATE_SIZE> >* outJacobians) const = 0;
  virtual void appendJacobians(const Eigen::Matrix<double,4,INNER_DIM>& lhs,
                               EvalTreeNode<TYPE>* evaluationTree,
                               std::vector<Jacobian<4,MAX_STATE_SIZE> >* outJacobians) const = 0;
  virtual void appendJacobians(const Eigen::Matrix<double,6,INNER_DIM>& lhs,
                               EvalTreeNode<TYPE>* evaluationTree,
                               std::vector<Jacobian<6,MAX_STATE_SIZE> >* outJacobians) const = 0;
};

} // steam

#include <steam/evaluator/BlockAutomaticEvaluator-inl.hpp>

#endif // STEAM_BLOCK_AUTOMATIC_EVALUATOR_HPP
