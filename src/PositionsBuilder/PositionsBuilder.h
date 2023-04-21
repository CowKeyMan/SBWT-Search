#ifndef POSITIONS_BUILDER_H
#define POSITIONS_BUILDER_H

/**
 * @file PositionsBuilder.h
 * @brief Builds a vector of the positions of the starting chracter of each kmer
 * in our sequence
 */

#include <cstddef>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::vector;

class PositionsBuilder {
private:
  u64 kmer_size;

public:
  explicit PositionsBuilder(u64 kmer_size);
  void build_positions(
    const vector<u64> &chars_before_newline,
    const u64 &string_size,
    vector<u64> &positions
  );

private:
  void process_one_string(
    u64 start_position_index,
    u64 end_position_index,
    u64 first_position,
    vector<u64> &positions
  );
};

}  // namespace sbwt_search
#endif
