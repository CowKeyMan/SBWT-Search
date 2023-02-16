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
  u64 read_padding_,
  u64 buffer_size_
):
    IndexFileParser(std::move(in_stream_), max_indexes_, read_padding_),
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
  shared_ptr<IndexesBatch> indexes_batch_,
  shared_ptr<IndexesStartsBatch> indexes_starts_batch_
) -> bool {
  u64 i = 0;
  IndexFileParser::generate_batch(
    std::move(indexes_batch_), std::move(indexes_starts_batch_)
  );
  const u64 initial_size = get_starts().size() + get_indexes().size();
  while (get_indexes().size() < get_max_indexes()
         && (!get_istream().eof() || buffer_index != buffer_size)) {
    i = get_next();
    if (buffer_size == 0) { break; }  // EOF
    if (new_read) {
      get_starts().push_back(get_indexes().size());
      new_read = false;
    }
    if (i == static_cast<u64>(-1) || i == static_cast<u64>(-2)) {
      ++get_indexes_batch()->skipped;
    } else if (i == static_cast<u64>(-3)) {
      pad_read();
      new_read = true;
    } else {
      ++get_indexes_batch()->true_indexes;
      get_indexes().push_back(i);
    }
  }
  pad_read();
  return get_starts().size() + get_starts().size() > initial_size;
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
