#include <memory>

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

ContinuousIndexFileParser::ContinuousIndexFileParser(
  u64 max_indexes_per_batch_,
  u64 max_batches,
  vector<string> filenames_,
  u64 read_padding_
):
    max_indexes_per_batch(max_indexes_per_batch_),
    read_padding(read_padding_),
    indexes_batch_producer(
      make_shared<IndexesBatchProducer>(max_indexes_per_batch_, max_batches)
    ),
    indexes_starts_batch_producer(
      make_shared<IndexesStartsBatchProducer>(max_batches)
    ),
    indexes_before_newfile_batch_producer(
      make_shared<IndexesBeforeNewfileBatchProducer>(max_batches)
    ),
    filenames(std::move(filenames_)) {
  filename_iterator = filenames.begin();
}

auto ContinuousIndexFileParser::read_and_generate() -> void {
  start_next_file();
  while (!fail) {
    do_at_batch_start();
    read_next();
    do_at_batch_finish();
  }
  do_at_generate_finish();
}

auto ContinuousIndexFileParser::start_next_file() -> bool {
  while (filename_iterator != filenames.end()) {
    indexes_before_newfile_batch_producer->add(
      indexes_batch_producer->current_write()->indexes.size()
    );
    auto filename = *filename_iterator++;
    try {
      Logger::log(
        Logger::LOG_LEVEL::INFO, format("Now reading file {}", filename)
      );
      open_parser(filename);
      return true;
    } catch (ios::failure &e) {
      Logger::log(Logger::LOG_LEVEL::ERROR, e.what());
    }
  }
  fail = true;
  return false;
}

auto ContinuousIndexFileParser::open_parser(const string &filename) -> void {
  auto in_stream = make_shared<ThrowingIfstream>(filename, ios::in);
  const string file_format = in_stream->read_string_with_size();
  if (file_format == "ascii") {
    index_file_parser = make_unique<AsciiIndexFileParser>(
      std::move(in_stream), max_indexes_per_batch, read_padding
    );
  } else if (file_format == "binary") {
    index_file_parser = make_unique<BinaryIndexFileParser>(
      std::move(in_stream), max_indexes_per_batch, read_padding
    );
  }
}

auto ContinuousIndexFileParser::do_at_batch_start() -> void {
  indexes_batch_producer->do_at_batch_start();
  indexes_starts_batch_producer->do_at_batch_start();
  indexes_before_newfile_batch_producer->do_at_batch_start();
  Logger::log_timed_event(
    "ContinuousIndexFileParser",
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  reset();
}

auto ContinuousIndexFileParser::reset() -> void {
  indexes_batch_producer->current_write()->reset();
  indexes_starts_batch_producer->current_write()->reset();
  indexes_before_newfile_batch_producer->current_write()->reset();
}

auto ContinuousIndexFileParser::read_next() -> void {
  while (
    (indexes_batch_producer->current_write()->indexes.size() < max_indexes_per_batch)
     && (
       index_file_parser->generate_batch(
         indexes_batch_producer->current_write(),
         indexes_starts_batch_producer->current_write()
       )
       || start_next_file()
      )
   ) {}
}

auto ContinuousIndexFileParser::do_at_batch_finish() -> void {
  auto indexes_batch = indexes_batch_producer->get_current_write();
  auto indexes_starts
    = indexes_starts_batch_producer->get_current_write()->indexes_starts;
  u64 reads = indexes_starts.size()
    + static_cast<u64>(indexes_starts.empty() || indexes_starts.front() != 0);
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format(
      "Read {} indexes in batch {}, of which {} are true indexes "
      "and {} is padding, and {} indexes were skipped because they represent "
      "nulls",
      indexes_batch->indexes.size(),
      reads,
      batch_id,
      indexes_batch->true_indexes,
      indexes_batch->indexes.size() - indexes_batch->true_indexes,
      indexes_batch->skipped
    )
  );
  Logger::log_timed_event(
    "ContinuousIndexFileParser",
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
  ++batch_id;
  indexes_batch_producer->do_at_batch_finish();
  indexes_starts_batch_producer->do_at_batch_finish();
  indexes_before_newfile_batch_producer->do_at_batch_finish();
}

auto ContinuousIndexFileParser::do_at_generate_finish() -> void {
  indexes_batch_producer->do_at_generate_finish();
  indexes_starts_batch_producer->do_at_generate_finish();
  indexes_before_newfile_batch_producer->do_at_generate_finish();
}

auto ContinuousIndexFileParser::get_indexes_batch_producer() const
  -> const shared_ptr<IndexesBatchProducer> & {
  return indexes_batch_producer;
}
auto ContinuousIndexFileParser::get_indexes_starts_batch_producer() const
  -> const shared_ptr<IndexesStartsBatchProducer> & {
  return indexes_starts_batch_producer;
}
auto ContinuousIndexFileParser::get_indexes_before_newfile_batch_producer(
) const -> const shared_ptr<IndexesBeforeNewfileBatchProducer> & {
  return indexes_before_newfile_batch_producer;
}

}  // namespace sbwt_search
