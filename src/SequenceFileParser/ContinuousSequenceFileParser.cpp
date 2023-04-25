#include <algorithm>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <bits/iterator_concepts.h>

#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "SequenceFileParser/IntervalBatchProducer.h"
#include "SequenceFileParser/StringBreakBatchProducer.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"
#include "kseqpp_read.hpp"

namespace sbwt_search {

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

ContinuousSequenceFileParser::ContinuousSequenceFileParser(
  u64 stream_id_,
  const vector<string> &filenames_,
  u64 kmer_size_,
  u64 max_chars_per_batch_,
  u64 max_seqs_per_batch_,
  u64 string_sequence_batch_producer_max_batches,
  u64 string_break_batch_producer_max_batches,
  u64 interval_batch_producer_max_batches
):
    filenames(filenames_),
    kmer_size(kmer_size_),
    batches(std::max(
      {string_sequence_batch_producer_max_batches,
       string_break_batch_producer_max_batches,
       interval_batch_producer_max_batches}
    )),
    max_chars_per_batch(max_chars_per_batch_),
    max_seqs_per_batch(max_seqs_per_batch_),
    string_sequence_batch_producer(make_shared<StringSequenceBatchProducer>(
      string_sequence_batch_producer_max_batches
    )),
    string_break_batch_producer(make_shared<StringBreakBatchProducer>(
      string_break_batch_producer_max_batches
    )),
    interval_batch_producer(
      make_shared<IntervalBatchProducer>(interval_batch_producer_max_batches)
    ),
    stream_id(stream_id_) {
  filename_iterator = filenames.begin();
  for (unsigned int i = 0; i < batches.capacity(); ++i) {
    batches.set(i, make_shared<Seq>(max_chars_per_batch, max_seqs_per_batch));
  }
}

auto ContinuousSequenceFileParser::read_and_generate() -> void {
  start_next_file();
  while (!fail) {
    do_at_batch_start();
    read_next();
    do_at_batch_finish();
  }
  do_at_generate_finish();
}

auto ContinuousSequenceFileParser::reset_rec() -> void {
  auto rec = batches.current_write();
  auto prev_rec = batches.current_read();
  if (prev_rec->seqs.empty()) {
    rec->clear();
    return;
  }
  int amount_to_copy = 0;
  if (prev_rec->chars_before_new_seq.size() == 1) {
    amount_to_copy = min<int>(
      static_cast<int>(kmer_size - 1), static_cast<int>(prev_rec->seqs.size())
    );
  } else {
    auto &chars_before_newline = prev_rec->chars_before_new_seq;
    amount_to_copy = min<int>(
      static_cast<int>(kmer_size - 1),
      static_cast<int>(
        prev_rec->seqs.size()
        - chars_before_newline[chars_before_newline.size() - 2]
      )
    );
  }
  rec->seqs.resize(rec->max_chars);
  copy(
    prev_rec->seqs.end() - amount_to_copy,
    prev_rec->seqs.end(),
    rec->seqs.begin()
  );
  rec->seqs.resize(amount_to_copy);
  rec->chars_before_new_seq.resize(0);
}

auto ContinuousSequenceFileParser::start_next_file() -> bool {
  while (filename_iterator != filenames.end()) {
    interval_batch_producer->add_file_start(
      batches.current_write()->chars_before_new_seq.size()
    );
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
  fail = true;
  return false;
}

auto ContinuousSequenceFileParser::do_at_batch_start() -> void {
  string_sequence_batch_producer->do_at_batch_start();
  string_break_batch_producer->do_at_batch_start();
  interval_batch_producer->do_at_batch_start();
  Logger::log_timed_event(
    format("SequenceFileParser_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  batches.step_write();
  reset_rec();
}

auto ContinuousSequenceFileParser::read_next() -> void {
  auto rec = batches.current_write();
  while ((rec->seqs.size() < max_chars_per_batch)
         && (rec->chars_before_new_seq.size() < max_seqs_per_batch)
         && ((*stream) >> (*rec) || start_next_file())) {}
  string_sequence_batch_producer->set_string(rec->seqs);
  string_break_batch_producer->set(rec->chars_before_new_seq, rec->seqs.size());
  interval_batch_producer->set_chars_before_newline(rec->chars_before_new_seq);
}

auto ContinuousSequenceFileParser::do_at_batch_finish() -> void {
  batches.step_read();
  auto seq_size = batches.current_write()->seqs.size();
  auto &str_breaks = batches.current_write()->chars_before_new_seq;
  str_breaks.push_back(std::numeric_limits<u64>::max());
  auto strings_in_batch = str_breaks.size()
    + static_cast<u64>(!str_breaks.empty()
                       && str_breaks.back() != (seq_size - 1));
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format(
      "Batch {} in stream {} contains {} indexes in {} seqs",
      batch_id,
      stream_id,
      seq_size,
      strings_in_batch
    )
  );
  Logger::log_timed_event(
    format("SequenceFileParser_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
  ++batch_id;
  string_sequence_batch_producer->do_at_batch_finish();
  string_break_batch_producer->do_at_batch_finish();
  interval_batch_producer->do_at_batch_finish();
}

auto ContinuousSequenceFileParser::do_at_generate_finish() -> void {
  string_sequence_batch_producer->do_at_generate_finish();
  string_break_batch_producer->do_at_generate_finish();
  interval_batch_producer->do_at_generate_finish();
}

auto ContinuousSequenceFileParser::get_string_sequence_batch_producer() const
  -> const shared_ptr<StringSequenceBatchProducer> & {
  return string_sequence_batch_producer;
}
auto ContinuousSequenceFileParser::get_string_break_batch_producer() const
  -> const shared_ptr<StringBreakBatchProducer> & {
  return string_break_batch_producer;
}
auto ContinuousSequenceFileParser::get_interval_batch_producer() const
  -> const shared_ptr<IntervalBatchProducer> & {
  return interval_batch_producer;
}

}  // namespace sbwt_search
