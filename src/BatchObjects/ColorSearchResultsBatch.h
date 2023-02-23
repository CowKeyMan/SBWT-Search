#ifndef COLOR_SEARCH_RESULTS_BATCH_H
#define COLOR_SEARCH_RESULTS_BATCH_H

/**
 * @file ColorSearchResultsBatch.h
 * @brief A vector of colour counts
 */

#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class ColorSearchResultsBatch {
public:
  vector<u64> results;
  auto reset() -> void { results.resize(0); }
};

}  // namespace sbwt_search

#endif
