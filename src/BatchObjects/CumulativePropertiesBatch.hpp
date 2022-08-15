#ifndef CUMULATIVE_PROPERTIES_BATCH_HPP
#define CUMULATIVE_PROPERTIES_BATCH_HPP

#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

class CumulativePropertiesBatch {
  public:
    vector<u64> cumsum_positions_per_string;
    vector<u64> cumsum_string_lengths;
};

}

#endif
