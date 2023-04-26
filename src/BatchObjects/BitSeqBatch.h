#ifndef BIT_SEQ_BATCH_H
#define BIT_SEQ_BATCH_H

/**
 * @file BitSeqBatch.h
 * @brief Container for the bit sequences. These are the converted binary
 * versions from  the string representation
 */

#include "Tools/PinnedVector.h"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using gpu_utils::PinnedVector;

class BitSeqBatch {
public:
  PinnedVector<u64> bit_seq;
  explicit BitSeqBatch(u64 bit_seq_size): bit_seq(bit_seq_size) {}
};

}  // namespace sbwt_search

#endif
