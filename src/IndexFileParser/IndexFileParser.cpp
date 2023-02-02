#include <memory>

#include "IndexFileParser/IndexFileParser.h"

namespace sbwt_search {

IndexFileParser::IndexFileParser(
  shared_ptr<ThrowingIfstream> in_stream_,
  shared_ptr<IndexesBatch> indexes_,
  shared_ptr<IndexesStartsBatch> indexes_starts_batch_,
  size_t max_indexes_,
  size_t read_padding_
):
  in_stream(std::move(in_stream_)),
  indexes(std::move(indexes_)),
  indexes_starts_batch(std::move(indexes_starts_batch_)),
  max_indexes(max_indexes_),
  read_padding(read_padding_) {}

auto IndexFileParser::get_istream() const -> ThrowingIfstream & {
  return *in_stream;
}
auto IndexFileParser::get_indexes() const -> vector<u64> & {
  return indexes->indexes;
}
auto IndexFileParser::get_max_indexes() const -> u64 { return max_indexes; }
auto IndexFileParser::get_starts() const -> vector<u64> & {
  return indexes_starts_batch->indexes_starts;
}
auto IndexFileParser::get_read_padding() const -> u64 { return read_padding; }

}  // namespace sbwt_search
