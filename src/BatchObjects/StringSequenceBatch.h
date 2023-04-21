#ifndef STRING_SEQUENCE_BATCH_H
#define STRING_SEQUENCE_BATCH_H

/**
 * @file StringSequenceBatch.h
 * @brief Data class for the string sequence associated with this batch, which
 * is a simple pointer of a character vector
 */

#include <string>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class StringSequenceBatch {
public:
  const vector<char> *seq;
};

}  // namespace sbwt_search

#endif
