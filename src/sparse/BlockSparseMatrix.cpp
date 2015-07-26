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
BlockSparseMatrix::BlockSparseMatrix(bool square, bool symmetric) : square_(square), symmetric_(symmetric) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Square matrix constructor, symmetry is still optional
//////////////////////////////////////////////////////////////////////////////////////////////
BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blkSqSizes, bool symmetric) : square_(true), symmetric_(symmetric) {
  this->reset(blkSqSizes, blkSqSizes);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Rectangular matrix constructor
//////////////////////////////////////////////////////////////////////////////////////////////
BlockSparseMatrix::BlockSparseMatrix(const std::vector<unsigned int>& blkRowSizes, const std::vector<unsigned int>& blkColSizes) : square_(false), symmetric_(false) {
  this->reset(blkRowSizes, blkColSizes);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Resize and clear matrix
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::reset(const std::vector<unsigned int>& blkRowSizes, const std::vector<unsigned int>& blkColSizes) {

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

  scalarRowDim_ = 0;
  scalarColDim_ = 0;
  blkRowSizes_ = blkRowSizes;
  blkColSizes_ = blkColSizes;
  cumBlkRowSizes_.reserve(blkRowSizes_.size());
  cumBlkColSizes_.reserve(blkColSizes_.size());
  for (std::vector<unsigned int>::const_iterator it = blkRowSizes_.begin(); it != blkRowSizes_.end(); ++it) {
    if (*it <= 0) {
      throw std::invalid_argument("Tried to initialize a block row size of 0.");
    }
    cumBlkRowSizes_.push_back(scalarRowDim_);
    scalarRowDim_ += *it;
  }
  for (std::vector<unsigned int>::const_iterator it = blkColSizes_.begin(); it != blkColSizes_.end(); ++it) {
    if (*it <= 0) {
      throw std::invalid_argument("Tried to initialize a block column size of 0.");
    }
    cumBlkColSizes_.push_back(scalarColDim_);
    scalarColDim_ += *it;
  }

  // init data
  cols_.clear();
  cols_.resize(blkColSizes_.size());
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Resize and clear a square matrix
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::reset(const std::vector<unsigned int>& blkSizes) {
  if (!square_) {
    throw std::invalid_argument("Tried to reset non-square block matrix "
                                "with square reset function.");
  }
  this->reset(blkSizes, blkSizes);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Clear sparse entries, maintain size
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::clear() {
  for (unsigned int c = 0; c < blkColSizes_.size(); c++) {
    cols_[c].rows.clear();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get number of block rows
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int BlockSparseMatrix::getNumBlkRows() const {
  return blkRowSizes_.size();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get number of block columns
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int BlockSparseMatrix::getNumBlkCols() const {
  return blkColSizes_.size();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Adds the matrix to the block entry at index (r,c), block dim must match
//////////////////////////////////////////////////////////////////////////////////////////////
void BlockSparseMatrix::add(unsigned int r, unsigned int c, const Eigen::MatrixXd& m) {
  if (symmetric_ && r > c) {
    std::cout << "[STEAM WARN] Attempted to add to lower half of upper-symmetric, "
                 "block-sparse matrix: operation was ignored for efficiency." << std::endl;
    return;
  }
  if (r >= blkRowSizes_.size() ||
      c >= blkColSizes_.size()) {
    throw std::invalid_argument("Requested row or column indice did not fall in valid "
                                "range of existing block structure.");
  }
  if (m.rows() != (int)blkRowSizes_[r] ||
      m.cols() != (int)blkColSizes_[c]) {
    std::stringstream ss; ss << "Size of matrix did not align with block structure; row: "
                             << r << " col: " << c << " failed the check: "
                             << m.rows() << " == " << (int)blkRowSizes_[r] << " && "
                             << m.cols() << " == " << (int)blkColSizes_[c];
    throw std::invalid_argument(ss.str());
  }

  std::map<unsigned int, BlockRowEntry>::iterator it = cols_[c].rows.find(r);
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
  if (r >= blkRowSizes_.size() ||
      c >= blkColSizes_.size()) {
    throw std::invalid_argument("Requested row or column indice did not fall in valid "
                                "range of existing block structure.");
  }

  if (symmetric_ && r > c) {
    throw std::invalid_argument("Attempted to read lower half of upper-symmetric... "
                                "cannot return reference");
  }

  std::map<unsigned int, BlockRowEntry>::iterator it = cols_[c].rows.find(r);
  if (it == cols_[c].rows.end()) {
    throw std::invalid_argument("Tried to read entry that did not exist");
  }

  return it->second.data;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Convert to Eigen sparse matrix format
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::SparseMatrix<double> BlockSparseMatrix::toEigen() const {

  // Allocate sparse matrix and reserve memory for estimates number of non-zero (nnz) entries
  Eigen::SparseMatrix<double> mat(scalarRowDim_, scalarColDim_);
  mat.reserve(this->getNnzPerCol());

  // Iterate over block-sparse columns and rows
  for (unsigned int c = 0; c < blkColSizes_.size(); c++) {
    for(std::map<unsigned int, BlockRowEntry>::const_iterator it = cols_[c].rows.begin(); it != cols_[c].rows.end(); ++it) {
      unsigned int r = it->first;
      // Iterate over internal matrix dimensions
      // Eigen matrix storage is column-major, outer iterator should be over column first for speed
      for (unsigned int j = 0; j < blkColSizes_[c]; j++) {
        for (unsigned int i = 0; i < blkRowSizes_[r]; i++) {
          // Check if value is non-zero (there may some extra sparsity to exploit inside 'dense' block matrices
          double v_ij = it->second.data(i,j);
          if (fabs(v_ij) > 0.0) {
            mat.insert(cumBlkRowSizes_[r] + i, cumBlkColSizes_[c] + j) = v_ij;
          }
        }
      }
    }
  }
  mat.makeCompressed(); // optional... maybe not necessary
  return mat;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Gets the number of non-zero entries per column (helper for prealloc. Eigen Sparse)
//////////////////////////////////////////////////////////////////////////////////////////////
Eigen::VectorXi BlockSparseMatrix::getNnzPerCol() const {

  Eigen::VectorXi result = Eigen::VectorXi(scalarColDim_);
  for (unsigned int c = 0; c < blkColSizes_.size(); c++) {
    unsigned int nnz = 0;
    for(std::map<unsigned int, BlockRowEntry>::const_iterator it = cols_[c].rows.begin(); it != cols_[c].rows.end(); ++it) {
      nnz += it->second.data.rows();
    }
    result.block(cumBlkColSizes_[c],0,blkColSizes_[c],1) = Eigen::VectorXi::Constant(blkColSizes_[c],nnz);
  }
  return result;
}

} // steam
