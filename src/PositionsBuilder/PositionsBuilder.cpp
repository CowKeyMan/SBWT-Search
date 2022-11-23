#include <algorithm>
#include <vector>

#include "PositionsBuilder/PositionsBuilder.h"
#include "Utils/TypeDefinitions.h"

using std::max;
using std::vector;

namespace sbwt_search {

PositionsBuilder::PositionsBuilder(const uint _kmer_size):
    kmer_size(_kmer_size){};

auto PositionsBuilder::build_positions(
  const vector<size_t> &chars_before_newline,
  const size_t &string_size,
  vector<size_t> &positions
) -> void {
  positions.resize(string_size);
  size_t start_position_index = 0, end_position_index = 0;
  size_t first_string_index = 0;
  size_t string_position_index = 0;
  size_t previous_string_break = 0;
  for (int i = 0; i < chars_before_newline.size() + 1; ++i) {
    if (i > 0) {
      start_position_index = max(start_position_index, end_position_index);
      first_string_index = chars_before_newline[i - 1];
    }
    if (i == chars_before_newline.size()) {
      if (string_size > (kmer_size - 1 + first_string_index)) {
        end_position_index
          += string_size - (kmer_size - 1 + first_string_index);
      }
    } else {
      if (chars_before_newline[i] > (kmer_size - 1 + first_string_index)) {
        end_position_index
          += chars_before_newline[i] - (kmer_size - 1 + first_string_index);
      }
    }
    process_one_string(
      start_position_index, end_position_index, first_string_index, positions
    );
  }
  positions.resize(end_position_index);
}

auto PositionsBuilder::process_one_string(
  const size_t start_position_index,
  const size_t end_position_index,
  const size_t first_position_content,
  vector<u64> &positions
) -> void {
  if (start_position_index >= end_position_index) { return; }
#pragma omp simd
  for (size_t i = 0; i < end_position_index - start_position_index; ++i) {
    positions[start_position_index + i] = first_position_content + i;
  }
}

}  // namespace sbwt_search
