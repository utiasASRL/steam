//////////////////////////////////////////////////////////////////////////////////////////////
/// \file BlockSparseMatrix.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/sparse/BlockSparseMatrix.hpp>

#include <stdexcept>
#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor, matrix size must still be set before using
//////////////////////////////////////////////////////////////////////////////////////////////
BlockSparseMatrix::BlockSparseMatrix(bool square, bool symmetric)
  : square_(square), symmetric_(symmetric) {

  if (!square_ && symmetric_) {
    throw std::invalid_argument("Tried to construct a symmetric matrix that is not square.");
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Rectangular matrix constructor
//////////////////////////////////////////////////////////////////////////////////////////////
BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blkRowSizes,
                                     const std::vector<unsigned int>& blkColSizes)
  : square_(false), symmetric_(false) {

  this->reset(blkRowSizes, blkColSizes);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Square matrix constructor, symmetry is still optional
//////////////////////////////////////////////////////////////////////////////////////////////
BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blkSqSizes, bool symmetric)
  : square_(true), symmetric_(symmetric) {

  this->reset(blkSqSizes);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Resize and clear matrix
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::reset(const std::vector<unsigned int>& blkRowSizes,
                              const std::vector<unsigned int>& blkColSizes) {

  // Check inputs
  if (square_ && blkRowSizes.size() != blkColSizes.size()) {
    throw std::invalid_argument("Tried to initialize a square block matrix "
                                "with uneven row and column entries.");
  }
  if (blkColSizes.size() <= 0) {
    throw std::invalid_argument("Tried to initialize a block matrix with no column size.");
  }
  if (blkRowSizes.size() <= 0) {
    throw std::invalid_argument("Tried to initialize a block matrix with no row size.");
  }

  // Setup indexing
  indexing_ = BlockMatrixIndexing(blkRowSizes, blkColSizes);

  // Setup data structures
  cols_.clear();
  cols_.resize(indexing_.colIndexing().numEntries());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Resize and clear a square matrix
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::reset(const std::vector<unsigned int>& blkSqSizes) {

  // Check inputs
  if (!square_) {
    throw std::invalid_argument("Tried to reset non-square block matrix "
                                "with square reset function.");
  }
  if (blkSqSizes.size() <= 0) {
    throw std::invalid_argument("Tried to initialize a block matrix with no row size.");
  }

  // Setup indexing
  indexing_ = BlockMatrixIndexing(blkSqSizes);

  // Setup data structures
  cols_.clear();
  cols_.resize(indexing_.colIndexing().numEntries());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Clear sparse entries, maintain size
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::clear() {
  for (unsigned int c = 0; c < indexing_.colIndexing().numEntries(); c++) {
    cols_[c].rows.clear();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Keep the existing entries and sizes, but set them to zero
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::zero() {
  for (unsigned int c = 0; c < indexing_.colIndexing().numEntries(); c++) {
    for(std::map<unsigned int, BlockRowEntry>::iterator it = cols_[c].rows.begin();
        it != cols_[c].rows.end(); ++it) {
      it->second.data.setZero();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get indexing object
//////////////////////////////////////////////////////////////////////////////////////////////
const BlockMatrixIndexing& BlockSparseMatrix::getIndexing() const {
  return indexing_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds the matrix to the block entry at index (r,c), block dim must match
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {

  // Get references to indexing objects
  const BlockDimIndexing& blkRowIndexing = indexing_.rowIndexing();
  const BlockDimIndexing& blkColIndexing = indexing_.colIndexing();

  // Check that indexing is valid
  if (r >= blkRowIndexing.numEntries() ||
      c >= blkColIndexing.numEntries()) {
    throw std::invalid_argument("Requested row or column indice did not fall in valid "
                                "range of existing block structure.");
  }

  // If symmetric, check that we are indexing into upper-triangular portion
  if (symmetric_ && r > c) {
    std::cout << "[STEAM WARN] Attempted to add to lower half of upper-symmetric, "
                 "block-sparse matrix: operation was ignored for efficiency." << std::endl;
    return;
  }

  // Check that provided matrix is of the correct dimensions
  if (m.rows() != (int)blkRowIndexing.blkSizeAt(r) ||
      m.cols() != (int)blkColIndexing.blkSizeAt(c)) {

    std::stringstream ss; ss << "Size of matrix did not align with block structure; row: "
                             << r << " col: " << c << " failed the check: "
                             << m.rows() << " == " << (int)blkRowIndexing.blkSizeAt(r) << " && "
                             << m.cols() << " == " << (int)blkColIndexing.blkSizeAt(c);
    throw std::invalid_argument(ss.str());
  }

  // Find if row entry exists
  std::map<unsigned int, BlockRowEntry>::iterator it = cols_[c].rows.find(r);

  // Check if found, and create new entry, or add to existing one
  if (it == cols_[c].rows.end()) {
    cols_[c].rows[r].data = m;
  } else {
    it->second.data += m;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Returns a const reference to the block entry at index (r,c)
//////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::MatrixXd& BlockSparseMatrix::read(unsigned int r, unsigned int c) {

  // Check that indexing is valid
  if (r >= indexing_.rowIndexing().numEntries() ||
      c >= indexing_.colIndexing().numEntries()) {
    throw std::invalid_argument("Requested row or column indice did not fall in valid "
                                "range of existing block structure.");
  }

  // If symmetric, check that we are indexing into upper-triangular portion
  if (symmetric_ && r > c) {
    std::cout << "[STEAM WARN] Attempted to add to lower half of upper-symmetric, "
                 "block-sparse matrix: cannot return reference." << std::endl;
  }

  // Find if row entry exists
  std::map<unsigned int, BlockRowEntry>::iterator it = cols_[c].rows.find(r);

  // If it does not exist, throw exception
  if (it == cols_[c].rows.end()) {
    throw std::invalid_argument("Tried to read entry that did not exist");
  }

  // Return reference to data
  return it->second.data;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convert to Eigen sparse matrix format
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::SparseMatrix<double> BlockSparseMatrix::toEigen(bool getSubBlockSparsity) const {

  // Get references to indexing objects
  const BlockDimIndexing& blkRowIndexing = indexing_.rowIndexing();
  const BlockDimIndexing& blkColIndexing = indexing_.colIndexing();

  // Allocate sparse matrix and reserve memory for estimates number of non-zero (nnz) entries
  Eigen::SparseMatrix<double> mat(blkRowIndexing.scalarSize(), blkColIndexing.scalarSize());
  mat.reserve(this->getNnzPerCol());

  // Iterate over block-sparse columns and rows
  for (unsigned int c = 0; c < blkColIndexing.numEntries(); c++) {
    for(std::map<unsigned int, BlockRowEntry>::const_iterator it = cols_[c].rows.begin(); it != cols_[c].rows.end(); ++it) {

      // Get row index of block entry
      unsigned int r = it->first;

      // Iterate over internal matrix dimensions
      // Eigen matrix storage is column-major, outer iterator should be over column first for speed
      for (unsigned int j = 0; j < blkColIndexing.blkSizeAt(c); j++) {
        for (unsigned int i = 0; i < blkRowIndexing.blkSizeAt(r); i++) {

          // Get scalar element
          double v_ij = it->second.data(i,j);

          // Add entry to sparse matrix
          // ** The case where we do not add the element is when sub-block sparsity is enabled
          //    and an element is exactly zero
          if (fabs(v_ij) > 0.0 || !getSubBlockSparsity) {
            mat.insert(blkRowIndexing.cumSumAt(r) + i, blkColIndexing.cumSumAt(c) + j) = v_ij;
          }
        }
      }
    }
  }

  // (optional) Compress into compact format
  mat.makeCompressed();
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Gets the number of non-zero entries per scalar-column
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXi BlockSparseMatrix::getNnzPerCol() const {

  // Get references to indexing objects
  const BlockDimIndexing& blkColIndexing = indexing_.colIndexing();

  // Allocate vector of ints
  Eigen::VectorXi result = Eigen::VectorXi(blkColIndexing.scalarSize());

  // Iterate over columns and determine number of non-zero entries
  for (unsigned int c = 0; c < blkColIndexing.numEntries(); c++) {

    // Sum
    unsigned int nnz = 0;

    // Iterate over sparse row entries of column 'c'
    for(std::map<unsigned int, BlockRowEntry>::const_iterator it = cols_[c].rows.begin();
        it != cols_[c].rows.end(); ++it) {
      nnz += it->second.data.rows();
    }

    // Add to result
    result.block(blkColIndexing.cumSumAt(c), 0, blkColIndexing.blkSizeAt(c), 1) =
        Eigen::VectorXi::Constant(blkColIndexing.blkSizeAt(c), nnz);
  }

  return result;
}

} // steam
