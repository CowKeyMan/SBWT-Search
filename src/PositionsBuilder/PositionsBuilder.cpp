#include <memory>
#include <vector>

#include "PositionsBuilder/PositionsBuilder.h"
#include "Utils/TypeDefinitions.h"

using std::make_unique;
using std::move;
using std::unique_ptr;
using std::vector;

namespace sbwt_search {

PositionsBuilder::PositionsBuilder(const u64 total_positions):
    positions(make_unique<vector<u64>>(total_positions)) {}

auto PositionsBuilder::get_positions(
  vector<u64> &cumsum_positions_per_string,
  vector<u64> &cumsum_string_lengths,
  int kmer_size
) -> unique_ptr<vector<u64>> {
#pragma omp parallel for
  for (int i = 0; i < cumsum_string_lengths.size() - 1; ++i) {
    auto start_position_index = cumsum_positions_per_string[i];
    auto string_length
      = cumsum_string_lengths[i + 1] - cumsum_string_lengths[i];
    auto end_position_index = start_position_index + string_length - kmer_size + 1;
    auto first_position = cumsum_string_lengths[i];
    process_one_string(
      start_position_index, end_position_index, first_position
    );
  }
  return move(positions);
}

auto PositionsBuilder::process_one_string(
  u64 start_position_index, u64 end_position_index, u64 first_position
) -> void {
  for (int i = 0; i < end_position_index - start_position_index; ++i) {
    (*positions)[start_position_index + i] = first_position + i;
  }
}

}
