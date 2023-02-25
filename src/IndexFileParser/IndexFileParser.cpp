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
auto IndexFileParser::get_max_indexes() const -> u64 { return max_indexes; }
auto IndexFileParser::get_read_padding() const -> u64 { return read_padding; }

auto IndexFileParser::generate_batch(
  shared_ptr<ReadStatisticsBatch> read_statistics_batch_,
  shared_ptr<WarpsBeforeNewReadBatch> warps_before_new_read_batch_,
  shared_ptr<IndexesBatch> indexes_batch_
) -> bool {
  read_statistics_batch = std::move(read_statistics_batch_);
  warps_before_new_read_batch = std::move(warps_before_new_read_batch_);
  indexes_batch = std::move(indexes_batch_);
  return false;
}

auto IndexFileParser::pad_read() -> void {
  while (get_indexes().size() % get_read_padding() != 0) {
    get_indexes().push_back(pad);
  }
}

auto IndexFileParser::get_read_statistics_batch() const
  -> const shared_ptr<ReadStatisticsBatch> & {
  return read_statistics_batch;
}
auto IndexFileParser::get_warps_before_new_read_batch() const
  -> const shared_ptr<WarpsBeforeNewReadBatch> & {
  return warps_before_new_read_batch;
}
auto IndexFileParser::get_indexes_batch() const
  -> const shared_ptr<IndexesBatch> & {
  return indexes_batch;
}

auto IndexFileParser::begin_new_read() -> void {
  get_warps_before_new_read_batch()->warps_before_new_read->push_back(
    get_indexes().size()
  );
  read_statistics_batch->found_idxs.push_back(0);
  read_statistics_batch->invalid_idxs.push_back(0);
  read_statistics_batch->not_found_idxs.push_back(0);
}

}  // namespace sbwt_search
