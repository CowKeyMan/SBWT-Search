#include <algorithm>
#include <bit>
#include <limits>

#include <fmt/core.h>

#include "ResultsPrinter/BoolContinuousResultsPrinter.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using math_utils::round_up;
using std::bit_cast;
using std::min;
using std::numeric_limits;

BoolContinuousResultsPrinter::BoolContinuousResultsPrinter(
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer_,
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer_,
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer_,
  vector<string> &filenames_,
  u64 kmer_size_,
  u64 max_chars_per_batch
):
    results_producer(std::move(results_producer_)),
    interval_producer(std::move(interval_producer_)),
    invalid_chars_producer(std::move(invalid_chars_producer_)),
    filenames(filenames_),
    kmer_size(kmer_size_),
    batch(round_up<u64>(max_chars_per_batch, sizeof(u64)) / sizeof(u64)) {
  reset_working_bits();
}

auto BoolContinuousResultsPrinter::read_and_generate() -> void {
  current_filename = filenames.begin();
  if (current_filename == filenames.end()) { return; }
  do_start_next_file();
  for (uint batch_id = 0; get_batch(); ++batch_id) {
    Logger::log_timed_event(
      "ResultsPrinter", Logger::EVENT_STATE::START, format("batch {}", batch_id)
    );
    process_batch();
    Logger::log_timed_event(
      "ResultsPrinter", Logger::EVENT_STATE::STOP, format("batch {}", batch_id)
    );
  }
  do_at_file_end();
}

auto BoolContinuousResultsPrinter::get_batch() -> bool {
  return (static_cast<uint>(*interval_producer >> interval_batch)
          & static_cast<uint>(*invalid_chars_producer >> invalid_chars_batch)
          & static_cast<uint>(*results_producer >> results_batch))
    > 0;
}

auto BoolContinuousResultsPrinter::process_batch() -> void {
  chars_index = results_index = line_index = 0;
  chars_before_newline_index = 0;
  for (auto newlines_before_newfile : interval_batch->newlines_before_newfile) {
    process_file(newlines_before_newfile);
    if (results_index >= results_batch->results.size()) { return; }
    do_start_next_file();
  }
}

auto BoolContinuousResultsPrinter::process_file(size_t newlines_before_newfile)
  -> void {
  for (; line_index < newlines_before_newfile
       && results_index < results_batch->results.size();
       ++line_index) {
    auto chars_before_newline
      = (*interval_batch->chars_before_newline)[chars_before_newline_index];
    process_line(chars_before_newline);
    if (chars_index + kmer_size > chars_before_newline) { do_with_newline(); }
    chars_index
      = (*interval_batch->chars_before_newline)[chars_before_newline_index];
    ++chars_before_newline_index;
  }
}

auto BoolContinuousResultsPrinter::process_line(size_t chars_before_newline)
  -> void {
  if (chars_before_newline < kmer_size - 1) { return; }
  invalid_chars_left = get_invalid_chars_left_first_kmer();
  while (chars_index < chars_before_newline - (kmer_size - 1)
         && results_index < results_batch->results.size()) {
    if (invalid_chars_batch->invalid_chars[chars_index + kmer_size - 1] == 1) {
      invalid_chars_left = kmer_size;
    }
    process_result(
      results_batch->results[results_index],
      results_batch->results[results_index] != numeric_limits<size_t>::max(),
      invalid_chars_left == 0
    );
    if (invalid_chars_left > 0) { --invalid_chars_left; }
    ++results_index;
    ++chars_index;
  }
}

auto BoolContinuousResultsPrinter::get_invalid_chars_left_first_kmer()
  -> size_t {
  auto &invalid_chars = invalid_chars_batch->invalid_chars;
  auto limit = min(
    {chars_index + kmer_size,
     invalid_chars.size(),
     (*interval_batch->chars_before_newline)[chars_before_newline_index]}
  );
  if (limit <= chars_index) { return 0; }
  // Even though this is a backwards loop, it was benchmarked and found to be
  // faster than the equivalent of this function in a forward loop manner
  // (since it stops earlier)
  for (size_t i = limit; i > chars_index; --i) {
    if (invalid_chars[i - 1] == 1) { return i - chars_index; }
  }
  return 0;
}

auto BoolContinuousResultsPrinter::process_result(
  size_t result, bool found, bool valid
) -> void {
  if (!valid) {
    do_invalid_result();
  } else if (!found) {
    do_not_found_result();
  } else {
    do_result(result);
  }
}

auto BoolContinuousResultsPrinter::do_start_next_file() -> void {
  if (current_filename != filenames.begin()) { do_at_file_end(); }
  do_open_next_file();
  do_write_file_header();
  current_filename = next(current_filename);
}

auto BoolContinuousResultsPrinter::do_invalid_result() -> void { shift(); }
auto BoolContinuousResultsPrinter::do_not_found_result() -> void { shift(); }
// NOLINTNEXTLINE(misc-unused-parameters)
auto BoolContinuousResultsPrinter::do_result(size_t result) -> void {
  working_bits |= (1ULL << shift_bits);
  shift();
}
auto BoolContinuousResultsPrinter::do_with_newline() -> void {
  seq_size_stream->write(bit_cast<char *>(&working_seq_size), sizeof(u64));
}

auto BoolContinuousResultsPrinter::do_at_file_end() -> void {
  dump_working_bits();
  reset_working_bits();
}

auto BoolContinuousResultsPrinter::do_open_next_file() -> void {
  working_seq_size = 0;
  out_stream = make_unique<ThrowingOfstream>(
    *current_filename + do_get_extension(), ios::binary | ios::out
  );
  seq_size_stream = make_unique<ThrowingOfstream>(
    *current_filename + get_seq_sizes_extension(), ios::binary | ios::out
  );
}

auto BoolContinuousResultsPrinter::reset_working_bits() -> void {
  working_bits = 0;
  shift_bits = default_shift_bits;
}

auto BoolContinuousResultsPrinter::shift() -> void {
  ++working_seq_size;
  if (shift_bits == 0) {
    dump_working_bits();
    reset_working_bits();
    return;
  }
  --shift_bits;
}

auto BoolContinuousResultsPrinter::dump_working_bits() -> void {
  out_stream->write(bit_cast<char *>(&working_bits), sizeof(u64));
}

auto BoolContinuousResultsPrinter::do_write_file_header() -> void {
  out_stream->write_string_with_size(do_get_format());
  out_stream->write_string_with_size(do_get_version());
  seq_size_stream->write_string_with_size(get_seq_sizes_format());
  seq_size_stream->write_string_with_size(get_seq_sizes_version());
}

auto BoolContinuousResultsPrinter::do_get_extension() -> string {
  return ".bool";
}
auto BoolContinuousResultsPrinter::get_seq_sizes_extension() -> string {
  return ".seqsizes";
}
auto BoolContinuousResultsPrinter::do_get_format() -> string { return "bool"; }
auto BoolContinuousResultsPrinter::do_get_version() -> string { return "v1.0"; }
auto BoolContinuousResultsPrinter::get_seq_sizes_format() -> string {
  return "seqsizes";
}
auto BoolContinuousResultsPrinter::get_seq_sizes_version() -> string {
  return "v1.0";
}

}  // namespace sbwt_search
