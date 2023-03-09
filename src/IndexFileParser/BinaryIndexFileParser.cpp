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
  u64 max_reads_,
  u64 warp_size_,
  u64 buffer_size_
):
    IndexFileParser(
      std::move(in_stream_), max_indexes_, max_reads_, warp_size_
    ),
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
  shared_ptr<ReadStatisticsBatch> read_statistics_batch_,
  shared_ptr<WarpsBeforeNewReadBatch> warps_before_new_read_batch_,
  shared_ptr<IndexesBatch> indexes_batch_
) -> bool {
  u64 i = 0;
  IndexFileParser::generate_batch(
    std::move(read_statistics_batch_),
    std::move(warps_before_new_read_batch_),
    std::move(indexes_batch_)
  );
  const u64 initial_size = get_indexes_batch()->indexes.size()
    + get_warps_before_new_read_batch()->warps_before_new_read->size();
  while (get_indexes().size() < get_max_indexes()
         && get_read_statistics_batch()->found_idxs.size() < get_max_reads()
         && (!get_istream().eof() || buffer_index != buffer_size)) {
    i = get_next();
    if (buffer_size == 0) { break; }  // EOF
    if (new_read) {
      begin_new_read();
      new_read = false;
    }
    if (i == static_cast<u64>(-1)) {
      ++get_read_statistics_batch()->not_found_idxs.back();
    } else if (i == static_cast<u64>(-2)) {
      ++get_read_statistics_batch()->invalid_idxs.back();
    } else if (i == static_cast<u64>(-3)) {
      pad_read();
      new_read = true;
    } else {
      ++get_read_statistics_batch()->found_idxs.back();
      get_indexes().push_back(i);
    }
  }
  pad_read();
  return (get_indexes_batch()->indexes.size()
          + get_warps_before_new_read_batch()->warps_before_new_read->size())
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
