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
    const uint kmer_size;

  public:
    PositionsBuilder(const uint kmer_size): kmer_size(kmer_size){};

    void build_positions(
      const vector<u64> &cumsum_positions_per_string,
      const vector<u64> &cumsum_string_lengths,
      vector<u64> &positions
    );

  private:
    void process_one_string(
      const u64 start_position_index,
      const u64 end_position_index,
      const u64 first_position,
      vector<u64> &positions
    );
};

}
#endif
