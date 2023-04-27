#ifndef INTERVAL_BATCH_H
#define INTERVAL_BATCH_H

/**
 * @file IntervalBatch.h
 * @brief Data class for the intervals. The chars_before_newline is a vector
 * which gives the number of characters that a string contains, before the new
 * one ends. newlines_before_newfile is a vector which
 * tells us how many newlines we need before we need to start considering the
 * next lines or sequences as originating from a new file. Note: the last
 * character of both vectores will always be the max value (ULLONG_MAX), and
 * both vectors are cumulative
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class IntervalBatch {
public:
  const vector<u64> *chars_before_new_seq;
  vector<u64> seqs_before_newfile;
};

}  // namespace sbwt_search

#endif
