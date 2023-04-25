#include <memory>

#include "IndexFileParser/IndexFileParser.h"

namespace sbwt_search {

IndexFileParser::IndexFileParser(
  shared_ptr<ThrowingIfstream> in_stream_,
  u64 max_indexes_,
  u64 max_seqs_,
  u64 warp_size_
):
    in_stream(std::move(in_stream_)),
    max_indexes(max_indexes_),
    max_seqs(max_seqs_),
    warp_size(warp_size_) {}

auto IndexFileParser::get_istream() const -> ThrowingIfstream & {
  return *in_stream;
}
auto IndexFileParser::get_indexes() const -> vector<u64> & {
  return indexes_batch->warped_indexes;
}
auto IndexFileParser::get_max_indexes() const -> u64 { return max_indexes; }
auto IndexFileParser::get_max_seqs() const -> u64 { return max_seqs; }

auto IndexFileParser::generate_batch(
  shared_ptr<SeqStatisticsBatch> read_statistics_batch_,
  shared_ptr<IndexesBatch> indexes_batch_
) -> bool {
  seq_statistics_batch = std::move(read_statistics_batch_);
  indexes_batch = std::move(indexes_batch_);
  return false;
}

auto IndexFileParser::end_seq() -> void {
  pad_warp();
  add_warp_interval();
  begin_new_seq();
}

auto IndexFileParser::get_num_seqs() -> u64 {
  return seq_statistics_batch->colored_seq_id.size();
}

auto IndexFileParser::pad_warp() -> void {
  while (get_indexes().size() % warp_size != 0) {
    get_indexes().push_back(pad);
  }
}

auto IndexFileParser::get_seq_statistics_batch() const
  -> const shared_ptr<SeqStatisticsBatch> & {
  return seq_statistics_batch;
}
auto IndexFileParser::get_indexes_batch() const
  -> const shared_ptr<IndexesBatch> & {
  return indexes_batch;
}

auto IndexFileParser::begin_new_seq() -> void {
  seq_statistics_batch->found_idxs.push_back(0);
  seq_statistics_batch->invalid_idxs.push_back(0);
  seq_statistics_batch->not_found_idxs.push_back(0);
  seq_statistics_batch->colored_seq_id.push_back(0);
}

auto IndexFileParser::add_warp_interval() -> void {
  auto &warp_intervals = indexes_batch->warps_intervals;
  seq_statistics_batch->colored_seq_id.back() = warp_intervals.size() - 1;
  if (warp_intervals.back() != indexes_batch->warped_indexes.size() / warp_size) {
    warp_intervals.push_back(indexes_batch->warped_indexes.size() / warp_size);
  }
}

}  // namespace sbwt_search
