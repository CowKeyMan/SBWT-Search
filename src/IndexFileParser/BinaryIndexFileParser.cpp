#include <bit>
#include <ios>
#include <stdexcept>

#include "IndexFileParser/BinaryIndexFileParser.h"

namespace sbwt_search {

using std::bit_cast;
using std::runtime_error;

BinaryIndexFileParser::BinaryIndexFileParser(
  shared_ptr<ThrowingIfstream> in_stream_,
  u64 max_indexes_,
  u64 max_seqs_,
  u64 warp_size_,
  u64 buffer_size_
):
    IndexFileParser(std::move(in_stream_), max_indexes_, max_seqs_, warp_size_),
    buffer_size(buffer_size_ * sizeof(u64)) {
  assert_version();
  buffer.resize(buffer_size);
  load_buffer();
}

auto BinaryIndexFileParser::assert_version() -> void {
  auto version = get_istream().read_string_with_size();
  if (version != "v1.0") {
    throw runtime_error("The file has an incompatible version number");
  }
}

auto BinaryIndexFileParser::generate_batch(
  shared_ptr<SeqStatisticsBatch> seq_statistics_batch_,
  shared_ptr<IndexesBatch> indexes_batch_
) -> bool {
  u64 i = 0;
  IndexFileParser::generate_batch(
    std::move(seq_statistics_batch_), std::move(indexes_batch_)
  );
  const u64 initial_size = get_indexes_batch()->warped_indexes.size()
    + get_seq_statistics_batch()->colored_seq_id.size();
  while (get_indexes().size() < get_max_indexes()
         && get_num_seqs() < get_max_seqs()
         && (!get_istream().eof() || buffer_index != buffer_size)) {
    i = get_next();
    if (buffer_size == 0) { break; }  // EOF
    if (i == static_cast<u64>(-1)) {
      ++get_seq_statistics_batch()->not_found_idxs.back();
    } else if (i == static_cast<u64>(-2)) {
      ++get_seq_statistics_batch()->invalid_idxs.back();
    } else if (i == static_cast<u64>(-3)) {
      end_seq();
    } else {
      ++get_seq_statistics_batch()->found_idxs.back();
      get_indexes().push_back(i);
    }
  }
  add_warp_interval();
  return (get_indexes_batch()->warped_indexes.size()
          + get_seq_statistics_batch()->colored_seq_id.size())
    > initial_size;
}

inline auto BinaryIndexFileParser::get_next() -> u64 {
  if (buffer_index >= buffer_size) { load_buffer(); }
  if (buffer_size == 0) { return -1; }
  return buffer[buffer_index++];
}

inline auto BinaryIndexFileParser::load_buffer() -> void {
  get_istream().read(
    bit_cast<char *>(buffer.data()),
    static_cast<std::streamsize>(buffer_size * sizeof(u64))
  );
  buffer_size = get_istream().gcount() / sizeof(u64);
  buffer_index = 0;
}

}  // namespace sbwt_search
