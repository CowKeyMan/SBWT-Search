#ifndef STRING_BREAK_BATCH_H
#define STRING_BREAK_BATCH_H

/**
 * @file StringBreakBatch.h
 * @brief Data which contains data about the points where strings end and
 * another starts
 */

#include <vector>

namespace sbwt_search {

using std::size_t;
using std::vector;

class StringBreakBatch {
public:
  const vector<size_t> *chars_before_newline;
  size_t string_size;
};

}  // namespace sbwt_search

#endif
