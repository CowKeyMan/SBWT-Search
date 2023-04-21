#ifndef STRING_BREAK_BATCH_H
#define STRING_BREAK_BATCH_H

/**
 * @file StringBreakBatch.h
 * @brief Data which contains data about the points where strings end and
 * another starts, as well as the string size which is how long the current
 * character vector associated with this batch is. Note that
 * chars_before_newline is cumulative
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class StringBreakBatch {
public:
  const vector<u64> *chars_before_newline;
  u64 string_size;
};

}  // namespace sbwt_search

#endif
