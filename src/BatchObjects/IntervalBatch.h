#ifndef INTERVAL_BATCH_H
#define INTERVAL_BATCH_H

/**
 * @file IntervalBatch.h
 * @brief Data class for the intervals. The string_lengths is a vector which
 * gives the length of each string in the current batch, which can even be 0.
 * strings_before_newfile is a vector which tells us how many strings we need
 * before we need to insert a linefeed symbol. If the current batch ends before
 * the file is finished, then the last entry in this is ULLONG_MAX. The last
 * entry is in fact always ULLONG_MAX, since when printing we use the number of
 * strings to know that we are done printing the current batch.
 */

#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

class IntervalBatch {
  public:
    vector<u64> string_lengths, strings_before_newfile;
};

}  // namespace sbwt_search

#endif
