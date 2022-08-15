#include <memory>
#include <vector>

#include "PositionsBuilder/PositionsBuilder.h"
#include "Utils/TypeDefinitions.h"

using std::make_unique;
using std::move;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

auto PositionsBuilder::build_positions(
  const vector<u64> &cumsum_positions_per_string,
  const vector<u64> &cumsum_string_lengths,
  vector<u64> &positions
) -> void {
  positions.resize(cumsum_positions_per_string.back());
#pragma omp parallel for
  for (int i = 0; i < cumsum_string_lengths.size() - 1; ++i) {
    auto start_position_index = cumsum_positions_per_string[i];
    auto string_length
      = cumsum_string_lengths[i + 1] - cumsum_string_lengths[i];
    auto end_position_index
      = start_position_index + string_length - kmer_size + 1;
    auto first_position = cumsum_string_lengths[i];
    process_one_string(
      start_position_index, end_position_index, first_position, positions
    );
  }
}

auto PositionsBuilder::process_one_string(
  const u64 start_position_index,
  const u64 end_position_index,
  const u64 first_position,
  vector<u64> &positions
) -> void {
  if (start_position_index > end_position_index) { return; }
  for (uint i = 0; i < end_position_index - start_position_index; ++i) {
    positions[start_position_index + i] = first_position + i;
  }
}

}
