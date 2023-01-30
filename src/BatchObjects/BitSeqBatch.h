#ifndef BIT_SEQ_BATCH_H
#define BIT_SEQ_BATCH_H

/**
 * @file BitSeqBatch.h
 * @brief Container for the bit sequences. These are the converted binary
 * versions from  the string representation
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class BitSeqBatch {
public:
  vector<u64> bit_seq;
};

}  // namespace sbwt_search

#endif
