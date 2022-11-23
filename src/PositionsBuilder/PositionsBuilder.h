#ifndef POSITIONS_BUILDER_H
#define POSITIONS_BUILDER_H

/**
 * @file PositionsBuilder.h
 * @brief Builds the positions of the valid bit sequences
 * */

#include <vector>

#include "Utils/TypeDefinitions.h"

using std::vector;

namespace sbwt_search {

class PositionsBuilder {
  private:
    const uint kmer_size;

  public:
    PositionsBuilder(const uint kmer_size);
    void build_positions(
      const vector<size_t> &chars_before_newline,
      const size_t &string_size,
      vector<size_t> &positions
    );

  private:
    void process_one_string(
      const size_t start_position_index,
      const size_t end_position_index,
      const size_t first_position,
      vector<size_t> &positions
    );
};

}  // namespace sbwt_search
#endif
