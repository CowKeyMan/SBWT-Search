#ifndef CONTINUOUS_RESULTS_PRINTER_HPP
#define CONTINUOUS_RESULTS_PRINTER_HPP

/**
 * @file ContinuousResultsPrinter.hpp
 * @brief Gets results, intervals and list of invalid chars and prints
 * these out to disk based on the given data and filenames.
 */

#include <algorithm>
#include <bit>
#include <memory>

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/InvalidCharsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "BatchObjects/StringSequenceBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using fmt::format;
using io_utils::ThrowingOfstream;
using log_utils::Logger;
using std::bit_cast;
using std::ios_base;
using std::make_unique;
using std::min;
using std::next;
using std::numeric_limits;
using std::shared_ptr;
using std::unique_ptr;

template <class TImplementation>
class ContinuousResultsPrinter {
private:
  auto impl() -> TImplementation & {
    return static_cast<TImplementation &>(*this);
  }
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer;
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer;
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer;
  shared_ptr<IntervalBatch> interval_batch;
  vector<string> &filenames;
  size_t chars_index = 0, results_index = 0, line_index = 0;
  size_t invalid_chars_left = 0;
  size_t chars_before_newline_index = 0;
  vector<string>::iterator current_filename;
  shared_ptr<ResultsBatch> results_batch;
  shared_ptr<InvalidCharsBatch> invalid_chars_batch;
  unique_ptr<ThrowingOfstream> out_stream;
  uint kmer_size;

protected:
  [[nodiscard]] auto get_current_filename() const -> const string & {
    return *current_filename;
  }

public:
  ContinuousResultsPrinter(
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer_,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer_,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>>
      invalid_chars_producer_,
    vector<string> &_filenames,
    uint kmer_size
  ):
    results_producer(std::move(results_producer_)),
    interval_producer(std::move(interval_producer_)),
    invalid_chars_producer(std::move(invalid_chars_producer_)),
    filenames(_filenames),
    kmer_size(kmer_size) {}

  auto read_and_generate() -> void {
    current_filename = filenames.begin();
    if (current_filename == filenames.end()) { return; }
    impl().do_start_next_file();
    for (uint batch_id = 0; get_batch(); ++batch_id) {
      Logger::log_timed_event(
        "ResultsPrinter",
        Logger::EVENT_STATE::START,
        format("batch {}", batch_id)
      );
      process_batch();
      Logger::log_timed_event(
        "ResultsPrinter",
        Logger::EVENT_STATE::STOP,
        format("batch {}", batch_id)
      );
    }
    impl().do_at_file_end();
  }

private:
  auto get_batch() -> bool {
    return (static_cast<uint>(*interval_producer >> interval_batch)
            & static_cast<uint>(*invalid_chars_producer >> invalid_chars_batch)
            & static_cast<uint>(*results_producer >> results_batch))
      > 0;
  }

  auto process_batch() -> void {
    chars_index = results_index = line_index = 0;
    chars_before_newline_index = 0;
    for (auto newlines_before_newfile :
         interval_batch->newlines_before_newfile) {
      process_file(newlines_before_newfile);
      if (results_index >= results_batch->results.size()) { return; }
      impl().do_start_next_file();
    }
  }

  auto process_file(size_t newlines_before_newfile) -> void {
    for (; line_index < newlines_before_newfile
         && results_index < results_batch->results.size();
         ++line_index) {
      auto chars_before_newline
        = (*interval_batch->chars_before_newline)[chars_before_newline_index];
      process_line(chars_before_newline);
      if (chars_index + kmer_size > chars_before_newline) {
        impl().do_with_newline();
      }
      chars_index
        = (*interval_batch->chars_before_newline)[chars_before_newline_index];
      ++chars_before_newline_index;
    }
  }

  auto process_line(size_t chars_before_newline) {
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

  auto get_invalid_chars_left_first_kmer() -> size_t {
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

  auto process_result(size_t result, bool found, bool valid) {
    if (!valid) {
      impl().do_invalid_result();
    } else if (!found) {
      impl().do_not_found_result();
    } else {
      impl().do_result(result);
    }
  }

protected:
  auto do_start_next_file() -> void {
    if (current_filename != filenames.begin()) { impl().do_at_file_end(); }
    impl().do_open_next_file();
    impl().do_write_file_header();
    current_filename = next(current_filename);
  }

  auto do_open_next_file() -> void {
    out_stream = make_unique<ThrowingOfstream>(
      get_current_filename() + impl().do_get_extension(),
      ios_base::binary | ios_base::out
    );
  };

  auto do_write_file_header() -> void {
    write_string_with_size(get_ostream(), impl().do_get_format());
    write_string_with_size(get_ostream(), impl().do_get_version());
  }

  auto do_at_file_end() -> void{};

  auto do_get_extension() -> string { return ""; }
  auto do_get_format() -> string { return "format_goes_here"; }
  auto do_get_version() -> string { return "version_number_goes_here"; }

  auto get_ostream() -> ThrowingOfstream & { return *out_stream; }
};

}  // namespace sbwt_search
#endif
