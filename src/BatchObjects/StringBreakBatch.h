#ifndef STRING_BREAK_BATCH_H
#define STRING_BREAK_BATCH_H

/**
 * @file StringBreakBatch.h
 * @brief Data which contains data about the points where strings end and
 * another starts
 */

#include <vector>

using std::vector;

namespace sbwt_search {

class StringBreakBatch {
  public:
    const vector<size_t> *chars_before_newline;
    size_t string_size;
};

}  // namespace sbwt_search

#endif
