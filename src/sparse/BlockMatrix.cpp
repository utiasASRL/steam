//////////////////////////////////////////////////////////////////////////////////////////////
/// \file BlockMatrix.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/sparse/BlockMatrix.hpp>

#include <stdexcept>
#include <iostream>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Default constructor, matrix size must still be set before using
//////////////////////////////////////////////////////////////////////////////////////////////
BlockMatrix::BlockMatrix(bool square, bool symmetric) : BlockMatrixBase(square, symmetric) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Rectangular matrix constructor
//////////////////////////////////////////////////////////////////////////////////////////////
BlockMatrix::BlockMatrix(const std::vector<unsigned int>& blkRowSizes,
                         const std::vector<unsigned int>& blkColSizes)
  : BlockMatrixBase(blkRowSizes, blkColSizes) {

  // Setup data structures
  cols_.clear();
  cols_.resize(this->getIndexing().colIndexing().numEntries());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Square matrix constructor, symmetry is still optional
//////////////////////////////////////////////////////////////////////////////////////////////
BlockMatrix::BlockMatrix(const std::vector<unsigned int>& blkSqSizes, bool symmetric)
  : BlockMatrixBase(blkSqSizes, symmetric) {

  // Setup data structures
  cols_.clear();
  cols_.resize(this->getIndexing().colIndexing().numEntries());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Keep the existing entries and sizes, but set them to zero
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockMatrix::zero() {
  for (unsigned int c = 0; c < this->getIndexing().colIndexing().numEntries(); c++) {
    for(std::map<unsigned int, BlockRowEntry>::iterator it = cols_[c].rows.begin();
        it != cols_[c].rows.end(); ++it) {
      it->second.data.setZero();
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds the matrix to the block entry at index (r,c), block dim must match
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {

  // Get references to indexing objects
  const BlockDimIndexing& blkRowIndexing = this->getIndexing().rowIndexing();
  const BlockDimIndexing& blkColIndexing = this->getIndexing().colIndexing();

  // Check that indexing is valid
  if (r >= blkRowIndexing.numEntries() ||
      c >= blkColIndexing.numEntries()) {
    throw std::invalid_argument("Requested row or column indice did not fall in valid "
                                "range of existing block structure.");
  }

  // If symmetric, check that we are indexing into upper-triangular portion
  if (this->isSymmetric() && r > c) {
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
/// \brief Returns a reference to the value at (r,c), if it exists
///        *Note this throws an exception if matrix is symmetric and you request a lower
///         triangular entry. For read operations, use copyAt(r,c).
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd& BlockMatrix::at(unsigned int r, unsigned int c) {

  // Check that indexing is valid
  if (r >= this->getIndexing().rowIndexing().numEntries() ||
      c >= this->getIndexing().colIndexing().numEntries()) {
    throw std::invalid_argument("Requested row or column indice did not fall in valid "
                                "range of existing block structure.");
  }

  // If symmetric, check that we are indexing into upper-triangular portion
  if (this->isSymmetric() && r > c) {
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
/// \brief Returns a copy of the entry at index (r,c)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd BlockMatrix::copyAt(unsigned int r, unsigned int c) const {

  // Check that indexing is valid
  if (r >= this->getIndexing().rowIndexing().numEntries() ||
      c >= this->getIndexing().colIndexing().numEntries()) {
    throw std::invalid_argument("Requested row or column indice did not fall in valid "
                                "range of existing block structure.");
  }

  // If symmetric, check that we are indexing into upper-triangular portion
  if (this->isSymmetric() && r > c) {

    // Accessing lower triangle of symmetric matrix

    // Find if row entry exists
    std::map<unsigned int, BlockRowEntry>::const_iterator it = cols_[c].rows.find(r);

    // If it does not exist, throw exception
    if (it == cols_[c].rows.end()) {
      return Eigen::MatrixXd::Zero(this->getIndexing().rowIndexing().blkSizeAt(r),
                                   this->getIndexing().colIndexing().blkSizeAt(c));
    }

    // Return reference to data
    return it->second.data.transpose();

  } else {

    // Not symmetric OR accessing upper-triangle

    // Find if row entry exists
    std::map<unsigned int, BlockRowEntry>::const_iterator it = cols_[c].rows.find(r);

    // If it does not exist, throw exception
    if (it == cols_[c].rows.end()) {
      return Eigen::MatrixXd::Zero(this->getIndexing().rowIndexing().blkSizeAt(r),
                                   this->getIndexing().colIndexing().blkSizeAt(c));
    }

    // Return reference to data
    return it->second.data;
  }
}

} // steam
