//////////////////////////////////////////////////////////////////////////////////////////////
/// \file BlockMatrixHelpers.cpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#include <steam/blockmat/BlockMatrixHelpers.hpp>

#include <stdexcept>
#include <iostream>

namespace steam {


/////////////////////////////////////////////////////////////////////////////////////////////
/// BlockDimIndexing
/////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
BlockDimIndexing::BlockDimIndexing() {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
BlockDimIndexing::BlockDimIndexing(const std::vector<unsigned int>& blkSizes)
  : blkSizes_(blkSizes) {

  // Check input has entries
  if (blkSizes_.size() <= 0) {
    throw std::invalid_argument("Tried to initialize a block matrix with no size.");
  }

  // Initialize scalar size and cumulative entries
  unsigned int i = 0;
  scalarDim_ = 0;
  cumBlkSizes_.resize(blkSizes_.size());
  for (std::vector<unsigned int>::const_iterator it = blkSizes_.begin(); it != blkSizes_.end(); ++it) {

    // Check that each input has a valid size
    if (*it <= 0) {
      throw std::invalid_argument("Tried to initialize a block row size of 0.");
    }

    // Add up cumulative sizes
    cumBlkSizes_[i] = scalarDim_;
    scalarDim_ += *it;
    i++;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the vector of block sizes
//////////////////////////////////////////////////////////////////////////////////////////////
const std::vector<unsigned int>& BlockDimIndexing::blkSizes() const {
  return blkSizes_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get number of block entries
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int BlockDimIndexing::numEntries() const {
  return blkSizes_.size();
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the block size of an entry
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int BlockDimIndexing::blkSizeAt(unsigned int index) const {
  return blkSizes_.at(index);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get the cumulative block size at an index
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int BlockDimIndexing::cumSumAt(unsigned int index) const {
  return cumBlkSizes_.at(index);
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get scalar size
//////////////////////////////////////////////////////////////////////////////////////////////
unsigned int BlockDimIndexing::scalarSize() const {
  return scalarDim_;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/// BlockMatrixIndexing
/////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor
//////////////////////////////////////////////////////////////////////////////////////////////
BlockMatrixIndexing::BlockMatrixIndexing()
  : blkSizeSymmetric_(false) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Constructor for a block-size-symmetric matrix
//////////////////////////////////////////////////////////////////////////////////////////////
BlockMatrixIndexing::BlockMatrixIndexing(const std::vector<unsigned int>& blkSizes)
  : blkRowIndexing_(blkSizes), blkSizeSymmetric_(true) {
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Rectangular matrix constructor
//////////////////////////////////////////////////////////////////////////////////////////////
BlockMatrixIndexing::BlockMatrixIndexing(const std::vector<unsigned int>& blkRowSizes,
                    const std::vector<unsigned int>& blkColSizes)
  : blkRowIndexing_(blkRowSizes), blkColIndexing_(blkColSizes), blkSizeSymmetric_(false) {
}


//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get block-row indexing
//////////////////////////////////////////////////////////////////////////////////////////////
const BlockDimIndexing& BlockMatrixIndexing::rowIndexing() const {
  return blkRowIndexing_;
}

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Get block-column indexing
//////////////////////////////////////////////////////////////////////////////////////////////
const BlockDimIndexing& BlockMatrixIndexing::colIndexing() const {
  if (!blkSizeSymmetric_) {
    return blkColIndexing_;
  } else {
    return blkRowIndexing_;
  }
}


} // steam
