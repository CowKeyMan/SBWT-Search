#ifndef BIT_SEQ_BATCH_H
#define BIT_SEQ_BATCH_H

/**
 * @file BitSeqBatch.h
 * @brief Container for the bit sequences. These are the converted binary
 * versions from  the string representation
 * */

#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

class BitSeqBatch {
  public:
    vector<u64> bit_seq;
};

}  // namespace sbwt_search

#endif
