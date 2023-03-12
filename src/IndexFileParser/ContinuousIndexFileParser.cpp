#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>

#include <fmt/core.h>

#include "IndexFileParser/AsciiIndexFileParser.h"
#include "IndexFileParser/BinaryIndexFileParser.h"
#include "IndexFileParser/ContinuousIndexFileParser.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using std::ios;
using std::make_shared;
using std::numeric_limits;

ContinuousIndexFileParser::ContinuousIndexFileParser(
  u64 stream_id_,
  u64 max_batches,
  u64 max_indexes_per_batch_,
  u64 max_reads_per_batch_,
  u64 warp_size_,
  vector<string> filenames_
):
    warps_before_new_read(create_warps_before_new_read(max_batches)),
    max_indexes_per_batch(max_indexes_per_batch_),
    max_reads_per_batch(max_reads_per_batch_),
    warp_size(warp_size_),
    colors_interval_batch_producer(make_shared<ColorsIntervalBatchProducer>(
      max_batches, warps_before_new_read
    )),
    read_statistics_batch_producer(
      make_shared<ReadStatisticsBatchProducer>(max_batches)
    ),
    warps_before_new_read_batch_producer(
      make_shared<WarpsBeforeNewReadBatchProducer>(
        max_batches, warps_before_new_read
      )
    ),
    indexes_batch_producer(
      make_shared<IndexesBatchProducer>(max_indexes_per_batch_, max_batches)
    ),
    filenames(std::move(filenames_)),
    stream_id(stream_id_) {
  filename_iterator = filenames.begin();
}

auto ContinuousIndexFileParser::read_and_generate() -> void {
  start_next_file();
  while (!fail) {
    do_at_batch_start();
    reset_batches();
    read_next();
    do_at_batch_finish();
  }
  do_at_generate_finish();
}

auto ContinuousIndexFileParser::reset_batches() -> void {
  colors_interval_batch_producer->current_write()->reset();
  read_statistics_batch_producer->current_write()->reset();
  warps_before_new_read_batch_producer->current_write()->reset();
  indexes_batch_producer->current_write()->reset();
}

auto ContinuousIndexFileParser::start_next_file() -> bool {
  while (filename_iterator != filenames.end()) {
    auto filename = *filename_iterator++;
    Logger::log(
      Logger::LOG_LEVEL::INFO, format("Now reading file {}", filename)
    );
    auto colors_interval_batch
      = colors_interval_batch_producer->current_write();
    colors_interval_batch->reads_before_newfile.push_back(
      colors_interval_batch->warps_before_new_read->size() + 1
    );
    warps_before_new_read_batch_producer->current_write()
      ->warps_before_new_read->push_back(
        indexes_batch_producer->current_write()->indexes.size() / warp_size
      );
    read_statistics_batch_producer->current_write()->found_idxs.push_back(0);
    read_statistics_batch_producer->current_write()->invalid_idxs.push_back(0);
    read_statistics_batch_producer->current_write()->not_found_idxs.push_back(0
    );
    try {
      start_new_file(filename);
      return true;
    } catch (ios::failure &e) {
      Logger::log(Logger::LOG_LEVEL::ERROR, e.what());
    }
  }
  fail = true;
  return false;
}

auto ContinuousIndexFileParser::start_new_file(const string &filename) -> void {
  auto in_stream = make_shared<ThrowingIfstream>(filename, ios::in);
  const string file_format = in_stream->read_string_with_size();
  if (file_format == "ascii") {  // NOLINT (bugprone-branch-clone)
    index_file_parser = make_unique<AsciiIndexFileParser>(
      std::move(in_stream),
      max_indexes_per_batch,
      max_reads_per_batch,
      warp_size
    );
  } else if (file_format == "binary") {
    index_file_parser = make_unique<BinaryIndexFileParser>(
      std::move(in_stream),
      max_indexes_per_batch,
      max_reads_per_batch,
      warp_size
    );
  } else {
    Logger::log(
      Logger::LOG_LEVEL::WARN, "Invalid file format in file: " + filename
    );
  }
}

auto ContinuousIndexFileParser::do_at_batch_start() -> void {
  colors_interval_batch_producer->do_at_batch_start();
  read_statistics_batch_producer->do_at_batch_start();
  warps_before_new_read_batch_producer->do_at_batch_start();
  indexes_batch_producer->do_at_batch_start();
  Logger::log_timed_event(
    format("ContinuousIndexFileParser_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
}

auto ContinuousIndexFileParser::read_next() -> void {
  while (
    (indexes_batch_producer->current_write()->indexes.size() < max_indexes_per_batch)
     && (read_statistics_batch_producer->current_write()->found_idxs.size() < max_reads_per_batch)
     && (
       index_file_parser->generate_batch(
         read_statistics_batch_producer->current_write(),
         warps_before_new_read_batch_producer->current_write(),
         indexes_batch_producer->current_write()
       )
       || start_next_file()
      )
   ) {}
}

auto ContinuousIndexFileParser::do_at_batch_finish() -> void {
  auto num_indexes = indexes_batch_producer->current_write()->indexes.size();
  auto read_statistics_batch
    = get_read_statistics_batch_producer()->current_write();
  auto reads = read_statistics_batch->found_idxs.size();
  auto num_found_idxs = std::accumulate(
    get_read_statistics_batch_producer()->current_write()->found_idxs.begin(),
    get_read_statistics_batch_producer()->current_write()->found_idxs.end(),
    0UL
  );
  auto num_invalid_idxs = std::accumulate(
    read_statistics_batch->invalid_idxs.begin(),
    read_statistics_batch->invalid_idxs.end(),
    0UL
  );
  auto num_not_found_idxs = std::accumulate(
    read_statistics_batch->not_found_idxs.begin(),
    read_statistics_batch->not_found_idxs.end(),
    0UL
  );
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format(
      "Batch {} contains {} indexes in {} reads, of which {} are found indexes "
      "and {} is "
      "padding. {} indexes were skipped, of which {} is not found and {} is "
      "invalids.",
      batch_id,
      num_indexes,
      reads,
      num_found_idxs,
      num_indexes - num_found_idxs,
      num_not_found_idxs + num_invalid_idxs,
      num_not_found_idxs,
      num_invalid_idxs
    )
  );
  Logger::log_timed_event(
    format("ContinuousIndexFileParser_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
  ++batch_id;
  colors_interval_batch_producer->current_write()
    ->reads_before_newfile.push_back(numeric_limits<u64>::max());
  colors_interval_batch_producer->current_write()
    ->warps_before_new_read->push_back(numeric_limits<u64>::max());
  colors_interval_batch_producer->do_at_batch_finish();
  read_statistics_batch_producer->do_at_batch_finish();
  warps_before_new_read_batch_producer->do_at_batch_finish();
  indexes_batch_producer->do_at_batch_finish();
}

auto ContinuousIndexFileParser::do_at_generate_finish() -> void {
  colors_interval_batch_producer->do_at_generate_finish();
  read_statistics_batch_producer->do_at_generate_finish();
  warps_before_new_read_batch_producer->do_at_generate_finish();
  indexes_batch_producer->do_at_generate_finish();
}

auto ContinuousIndexFileParser::get_colors_interval_batch_producer() const
  -> const shared_ptr<ColorsIntervalBatchProducer> & {
  return colors_interval_batch_producer;
}
auto ContinuousIndexFileParser::get_read_statistics_batch_producer() const
  -> const shared_ptr<ReadStatisticsBatchProducer> & {
  return read_statistics_batch_producer;
}
auto ContinuousIndexFileParser::get_warps_before_new_read_batch_producer() const
  -> const shared_ptr<WarpsBeforeNewReadBatchProducer> & {
  return warps_before_new_read_batch_producer;
}
auto ContinuousIndexFileParser::get_indexes_batch_producer() const
  -> const shared_ptr<IndexesBatchProducer> & {
  return indexes_batch_producer;
}

auto ContinuousIndexFileParser::create_warps_before_new_read(u64 amount) const
  -> vector<shared_ptr<vector<u64>>> {
  vector<shared_ptr<vector<u64>>> result;
  for (int i = 0; i < amount; ++i) {
    result.push_back(make_shared<vector<u64>>());
  }
  return result;
}

}  // namespace sbwt_search
