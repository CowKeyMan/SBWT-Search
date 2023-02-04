#ifndef INDEXES_BEFORE_NEWFILE_BATCH_H
#define INDEXES_BEFORE_NEWFILE_BATCH_H

/**
 * @file IndexesBeforeNewfileBatch.h
 * @brief Holds a vector of integers which represent how many indexes belong
 * the current file before we need to start considering the next indexes to
 * belong to another file
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::size_t;
using std::vector;

class IndexesBeforeNewfileBatch {
public:
  vector<size_t> indexes_before_newfile;
  auto reset() -> void { indexes_before_newfile.resize(0); }
};

}  // namespace sbwt_search

#endif
