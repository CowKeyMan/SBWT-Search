#include <algorithm>
#include <vector>

#include "PositionsBuilder/PositionsBuilder.h"
#include "Tools/TypeDefinitions.h"

using std::max;
using std::vector;

namespace sbwt_search {

PositionsBuilder::PositionsBuilder(const u64 _kmer_size):
    kmer_size(_kmer_size){};

auto PositionsBuilder::build_positions(
  const vector<u64> &chars_before_newline,
  const u64 &string_size,
  vector<u64> &positions
) -> void {
  u64 start_position_index = 0, end_position_index = 0;
  u64 first_string_index = 0;
  u64 string_position_index = 0;
  u64 previous_string_break = 0;
  positions.resize(string_size);
  for (int i = 0; i < chars_before_newline.size(); ++i) {
    if (i > 0) {
      start_position_index = max(start_position_index, end_position_index);
      first_string_index = chars_before_newline[i - 1];
    }
    if (i == chars_before_newline.size() - 1) {
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
  const u64 start_position_index,
  const u64 end_position_index,
  const u64 first_position_content,
  vector<u64> &positions
) -> void {
  if (start_position_index >= end_position_index) { return; }
#pragma omp simd
  for (u64 i = 0; i < end_position_index - start_position_index; ++i) {
    positions[start_position_index + i] = first_position_content + i;
  }
}

}  // namespace sbwt_search
