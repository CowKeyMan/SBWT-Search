#include <ios>
#include <memory>
#include <string>
#include <vector>

#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "SequenceFileParser/IntervalBatchProducer.h"
#include "SequenceFileParser/StringBreakBatchProducer.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Utils/IOUtils.h"
#include "Utils/Logger.h"
#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"
#include "kseqpp_read.hpp"

using fmt::format;
using io_utils::ThrowingIfstream;
using log_utils::Logger;
using reklibpp::Seq;
using reklibpp::SeqStreamIn;
using std::ios;
using std::make_shared;
using std::make_unique;
using std::min;
using std::shared_ptr;
using std::string;
using std::vector;

namespace sbwt_search {

ContinuousSequenceFileParser::ContinuousSequenceFileParser(
  const vector<string> &_filenames,
  const uint _kmer_size,
  const size_t _max_chars_per_batch,
  const size_t max_batches,
  shared_ptr<StringSequenceBatchProducer> _string_sequence_batch_producer,
  shared_ptr<StringBreakBatchProducer> _string_break_batch_producer,
  shared_ptr<IntervalBatchProducer> _interval_batch_producer
):
    filenames(_filenames),
    kmer_size(_kmer_size),
    batches(max_batches),
    max_chars_per_batch(_max_chars_per_batch),
    string_sequence_batch_producer(_string_sequence_batch_producer),
    string_break_batch_producer(_string_break_batch_producer),
    interval_batch_producer(_interval_batch_producer) {
  filename_iterator = filenames.begin();
  for (unsigned int i = 0; i < batches.capacity(); ++i) {
    batches.set(i, make_shared<Seq>(Seq(max_chars_per_batch)));
  }
}

auto ContinuousSequenceFileParser::read_and_generate() -> void {
  start_next_file();
  while (!fail) {
    do_at_batch_start();
    batches.step_write();
    reset_rec();
    read_next();
    batches.step_read();
    do_at_batch_finish();
  }
  do_at_generate_finish();
}

auto ContinuousSequenceFileParser::reset_rec() -> void {
  auto rec = batches.current_write();
  auto prev_rec = batches.current_read();
  if (prev_rec->seq.size() == 0) {
    rec->clear();
    return;
  }
  size_t amount_to_copy;
  if (prev_rec->string_breaks.size() == 0) {
    amount_to_copy = min<size_t>(kmer_size - 1, prev_rec->seq.size());
  } else {
    amount_to_copy = min<size_t>(
      kmer_size - 1, prev_rec->seq.size() - prev_rec->string_breaks.back()
    );
  }
  rec->seq.resize(rec->max_seq_size);
  copy(
    prev_rec->seq.end() - amount_to_copy, prev_rec->seq.end(), rec->seq.begin()
  );
  rec->seq.resize(amount_to_copy);
  rec->string_breaks.resize(0);
}

auto ContinuousSequenceFileParser::start_next_file() -> bool {
  while (filename_iterator != filenames.end()) {
    interval_batch_producer->add_file_end(batches.current_write()->seq.size());
    auto filename = *filename_iterator++;
    try {
      ThrowingIfstream::check_file_exists(filename);
      Logger::log(
        Logger::LOG_LEVEL::INFO, format("Now reading file {}", filename)
      );
      stream = make_unique<SeqStreamIn>(filename.c_str());
      return true;
    } catch (ios::failure &e) {
      Logger::log(Logger::LOG_LEVEL::ERROR, e.what());
    }
  }
  interval_batch_producer->add_file_end(batches.current_write()->seq.size());
  fail = true;
  return false;
}

auto ContinuousSequenceFileParser::do_at_batch_start() -> void {
  string_sequence_batch_producer->do_at_batch_start();
  string_break_batch_producer->do_at_batch_start();
  interval_batch_producer->do_at_batch_start();
  Logger::log_timed_event(
    "SequenceFileParser",
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
}

auto ContinuousSequenceFileParser::read_next() -> void {
  auto rec = batches.current_write();
  while ((rec->seq.size() < max_chars_per_batch)
         && ((*stream) >> (*rec) || start_next_file())) {
    ++batch_id;
  }
  string_sequence_batch_producer->set_string(rec->seq);
  string_break_batch_producer->set(rec->string_breaks, rec->seq.size());
  interval_batch_producer->set_string_breaks(rec->string_breaks);
}

auto ContinuousSequenceFileParser::do_at_batch_finish() -> void {
  auto str_breaks = batches.current_write()->string_breaks;
  auto seq_size = batches.current_write()->seq.size();
  auto strings_in_batch
    = str_breaks.size()
    + (str_breaks.size() != 0 && str_breaks.back() != (seq_size - 1));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format(
      "Read {} characters from {} strings in batch {}",
      seq_size,
      strings_in_batch,
      batch_id
    )
  );
  Logger::log_timed_event(
    "SequenceFileParser",
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
  string_sequence_batch_producer->do_at_batch_finish();
  string_break_batch_producer->do_at_batch_finish();
  interval_batch_producer->do_at_batch_finish();
}

auto ContinuousSequenceFileParser::do_at_generate_finish() -> void {
  string_sequence_batch_producer->do_at_generate_finish();
  string_break_batch_producer->do_at_generate_finish();
  interval_batch_producer->do_at_generate_finish();
}

}  // namespace sbwt_search
