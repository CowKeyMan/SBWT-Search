#ifndef STRING_BREAK_BATCH_H
#define STRING_BREAK_BATCH_H

/**
 * @file StringBreakBatch.h
 * @brief String breaks are the last character for a string before it ends
 */

#include <vector>

using std::vector;

namespace sbwt_search {

class StringBreakBatch {
  public:
    const vector<size_t> *string_breaks;
    size_t string_size;
};

}  // namespace sbwt_search

#endif
