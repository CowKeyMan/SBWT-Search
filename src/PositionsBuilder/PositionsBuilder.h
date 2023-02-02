#ifndef POSITIONS_BUILDER_H
#define POSITIONS_BUILDER_H

/**
 * @file PositionsBuilder.h
 * @brief Builds the positions of the valid bit sequences
 */

#include <cstddef>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using std::size_t;
using std::vector;

class PositionsBuilder {
private:
  uint kmer_size;

public:
  explicit PositionsBuilder(uint kmer_size);
  void build_positions(
    const vector<size_t> &chars_before_newline,
    const size_t &string_size,
    vector<size_t> &positions
  );

private:
  void process_one_string(
    size_t start_position_index,
    size_t end_position_index,
    size_t first_position,
    vector<size_t> &positions
  );
};

}  // namespace sbwt_search
#endif
