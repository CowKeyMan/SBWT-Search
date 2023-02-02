#ifndef READS_BEFORE_NEWFILE_BATCH_H
#define READS_BEFORE_NEWFILE_BATCH_H

/**
 * @file ReadsBeforeNewfileBatch.h
 * @brief Holds a vector of integers which represent how many reads belong to
 * the current file before we need to start considering the next reads to belong
 * to another file
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class ReadsBeforeNewfileBatch {
public:
  vector<u64> reads_before_newfile;
  auto reset() -> void { reads_before_newfile.resize(0); }
};

}  // namespace sbwt_search

#endif
