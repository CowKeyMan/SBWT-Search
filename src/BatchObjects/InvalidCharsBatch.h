#ifndef INVALID_CHARS_BATCH_H
#define INVALID_CHARS_BATCH_H

/**
 * @file InvalidCharsBatch.h
 * @brief Contains a binary vector of wether a string is valid or not
 * */

#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

class InvalidCharsBatch {
  public:
    vector<char> invalid_chars;
};

}  // namespace sbwt_search

#endif
