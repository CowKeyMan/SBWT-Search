#ifndef POSITIONS_BUILDER_H
#define POSITIONS_BUILDER_H

/**
 * @file PositionsBuilder.hpp
 * @brief Builds the positions of the valid bit sequences
 * */

#include <memory>
#include <vector>

#include "Utils/TypeDefinitions.h"

using std::make_unique;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

class PositionsBuilder {
  private:
    unique_ptr<vector<u64>> positions;

  public:
    PositionsBuilder(){};

    unique_ptr<vector<u64>> get_positions(
      vector<u64> &cumsum_positions_per_string,
      vector<u64> &cumsum_string_lengths,
      int kmer_size
    );

  private:
    void process_one_string(
      u64 start_position_index, u64 end_position_index, u64 first_position
    );
};

}
#endif
