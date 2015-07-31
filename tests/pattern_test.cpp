#include "catch.hpp"

#include <iostream>

#include <steam/blockmat/BlockSparseMatrix.hpp>
#include <steam/blockmat/BlockVector.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////
/// Sample Test
/////////////////////////////////////////////////////////////////////////////////////////////
TEST_CASE("Test the use of sparsity patterns", "[pattern]" ) {

  std::vector<unsigned int> blockSizes;
  blockSizes.resize(3);
  blockSizes[0] = 2;
  blockSizes[1] = 2;
  blockSizes[2] = 2;

  // Setup blocks
  Eigen::Matrix2d m_tri_diag; m_tri_diag << 2, 1, 1, 2;
  Eigen::Matrix2d m_tri_offdiag; m_tri_offdiag << 0, 0, 1, 0;
  Eigen::Matrix2d m_z = Eigen::Matrix2d::Zero();
  Eigen::Matrix2d m_o = Eigen::Matrix2d::Ones();

  // Setup A - Case 1
  steam::BlockSparseMatrix tri(blockSizes, true);
  tri.add(0, 0, m_tri_diag);
  tri.add(0, 1, m_tri_offdiag);
  tri.add(1, 1, m_tri_diag);
  tri.add(1, 2, m_tri_offdiag);
  tri.add(2, 2, m_tri_diag);

  // Setup A - Case 2
  steam::BlockSparseMatrix blkdiag(blockSizes, true);
  blkdiag.add(0, 0, m_tri_diag);
  blkdiag.add(0, 1, m_z);
  blkdiag.add(1, 1, m_tri_diag);
  blkdiag.add(1, 2, m_z);
  blkdiag.add(2, 2, m_tri_diag);

  // Setup A - Case 3
  steam::BlockSparseMatrix tri_ones(blockSizes, true);
  tri_ones.add(0, 0, m_o);
  tri_ones.add(0, 1, m_o);
  tri_ones.add(1, 1, m_o);
  tri_ones.add(1, 2, m_o);
  tri_ones.add(2, 2, m_o);

  // Setup B
  Eigen::VectorXd b(6); b << 1, 2, 3, 4, 5, 6;

  SECTION("Test sub sparsity" ) {

    // Maximum sparsity
    Eigen::SparseMatrix<double> eig_sparse = tri.toEigen(true);
    INFO("case1: " << eig_sparse);
    INFO("nonzeros: " << eig_sparse.nonZeros());
    CHECK(eig_sparse.nonZeros() == 14);

    // Get only block-level sparsity (important for re-using pattern)
    Eigen::SparseMatrix<double> eig_blk_sparse = tri.toEigen(false);
    INFO("case2: " << eig_blk_sparse);
    INFO("nonzeros: " << eig_blk_sparse.nonZeros());
    CHECK(eig_blk_sparse.nonZeros() == 20);

  }

  SECTION("Test solve" ) {

    // Maximum sparsity
    Eigen::SparseMatrix<double> eig_sparse = tri.toEigen(true);
    INFO("case1: " << eig_sparse);
    INFO("nonzeros: " << eig_sparse.nonZeros());

    // Solve sparse
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.analyzePattern(eig_sparse);
    solver.factorize(eig_sparse);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x1 = solver.solve(b);
    INFO("x1: " << x1.transpose());

    // Get only block-level sparsity (important for re-using pattern)
    Eigen::SparseMatrix<double> eig_blk_sparse = tri.toEigen(false);
    INFO("case2: " << eig_blk_sparse);
    INFO("nonzeros: " << eig_blk_sparse.nonZeros());

    // Solve sparse
    solver.analyzePattern(eig_blk_sparse);
    solver.factorize(eig_blk_sparse);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x2 = solver.solve(b);
    INFO("x2: " << x2.transpose());
    CHECK((x1-x2).norm() < 1e-6);

  }

  SECTION("Test solve, setting pattern with ones" ) {

    // Solve using regular tri-block diagonal
    Eigen::SparseMatrix<double> eig_tri = tri.toEigen();
    INFO("case1: " << eig_tri);
    INFO("nonzeros: " << eig_tri.nonZeros());

    // Solve sparse
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.analyzePattern(eig_tri);
    solver.factorize(eig_tri);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x1 = solver.solve(b);
    INFO("x1: " << x1.transpose());

    // Set pattern using ones and then solve with tri-block
    Eigen::SparseMatrix<double> eig_tri_ones = tri_ones.toEigen();
    INFO("case2: " << eig_tri_ones);
    INFO("nonzeros: " << eig_tri_ones.nonZeros());

    // Solve sparse
    solver.analyzePattern(eig_tri_ones);
    solver.factorize(eig_tri);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x2 = solver.solve(b);
    INFO("x2: " << x2.transpose());
    CHECK((x1-x2).norm() < 1e-6);

  }

  SECTION("Test solve of matrix with zero blocks, setting pattern with ones" ) {

    // Solve using regular tri-block diagonal
    Eigen::SparseMatrix<double> eig_blkdiag = blkdiag.toEigen();
    INFO("case1: " << eig_blkdiag);
    INFO("nonzeros: " << eig_blkdiag.nonZeros());

    // Solve sparse
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
    solver.analyzePattern(eig_blkdiag);
    solver.factorize(eig_blkdiag);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x1 = solver.solve(b);
    INFO("x1: " << x1.transpose());

    // Set pattern using ones and then solve with tri-block
    Eigen::SparseMatrix<double> eig_tri_ones = tri_ones.toEigen();
    INFO("case2: " << eig_tri_ones);
    INFO("nonzeros: " << eig_tri_ones.nonZeros());

    // Solve sparse
    solver.analyzePattern(eig_tri_ones);
    solver.factorize(eig_blkdiag);
    if (solver.info() != Eigen::Success) {
      throw std::runtime_error("Decomp failure.");
    }
    Eigen::VectorXd x2 = solver.solve(b);
    INFO("x2: " << x2.transpose());
    CHECK((x1-x2).norm() < 1e-6);

  }


} // TEST_CASE
