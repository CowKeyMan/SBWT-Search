#include <memory>

#include "IndexFileParser/IndexFileParser.h"

namespace sbwt_search {

IndexFileParser::IndexFileParser(
  shared_ptr<ThrowingIfstream> in_stream_, u64 max_indexes_, u64 read_padding_
):
    in_stream(std::move(in_stream_)),
    max_indexes(max_indexes_),
    read_padding(read_padding_) {}

auto IndexFileParser::get_istream() const -> ThrowingIfstream & {
  return *in_stream;
}
auto IndexFileParser::get_indexes() const -> vector<u64> & {
  return indexes_batch->indexes;
}
auto IndexFileParser::get_indexes_batch() -> shared_ptr<IndexesBatch> & {
  return indexes_batch;
}
auto IndexFileParser::get_max_indexes() const -> u64 { return max_indexes; }
auto IndexFileParser::get_starts() const -> vector<u64> & {
  return indexes_starts_batch->indexes_starts;
}
auto IndexFileParser::get_read_padding() const -> u64 { return read_padding; }

auto IndexFileParser::generate_batch(
  shared_ptr<IndexesBatch> indexes_batch_,
  shared_ptr<IndexesStartsBatch> indexes_starts_batch_
) -> bool {
  indexes_batch = std::move(indexes_batch_);
  indexes_starts_batch = std::move(indexes_starts_batch_);
  return false;
}

auto IndexFileParser::pad_read() -> void {
  while (get_indexes().size() % get_read_padding() != 0) {
    get_indexes().push_back(pad);
  }
}

}  // namespace sbwt_search
