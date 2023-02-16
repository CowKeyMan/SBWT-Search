#ifndef INTERVAL_BATCH_H
#define INTERVAL_BATCH_H

/**
 * @file IntervalBatch.h
 * @brief Data class for the intervals. The chars_before_newline is a vector
 * which gives the number of characters that a string contains, before the new
 * one ends. This is cumulative. characters_before_newfile is a vector which
 * tells us how many characters we need before we need to insert a linefeed
 * symbol. Note: the last character will always be the max value (ULLONG_MAX)
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class IntervalBatch {
public:
  const vector<u64> *chars_before_newline;
  vector<u64> newlines_before_newfile;
};

}  // namespace sbwt_search

#endif
