#ifndef INTERVAL_BATCH_H
#define INTERVAL_BATCH_H

/**
 * @file IntervalBatch.h
 * @brief Data class for the intervals. The string_breaks is a vector which
 * gives the index of each last character for each string in the current batch.
 * characters_before_newfile is a vector which tells us how many characters we
 * need before we need to insert a linefeed symbol. Note: the last character
 * will always be the max value (ULLONG_MAX)
 * */

#include <vector>

using std::vector;

namespace sbwt_search {

class IntervalBatch {
  public:
    const vector<size_t> *string_breaks;
    vector<size_t> characters_before_newfile;
};

}  // namespace sbwt_search

#endif
