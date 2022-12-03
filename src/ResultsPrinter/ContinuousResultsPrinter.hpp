#ifndef CONTINUOUS_RESULTS_PRINTER_HPP
#define CONTINUOUS_RESULTS_PRINTER_HPP

/**
 * @file ContinuousResultsPrinter.hpp
 * @brief Gets results, intervals and list of invalid chars and prints
 * these out to disk based on the given data and filenames.
 * */

#include <algorithm>
#include <chrono>
#include <climits>
#include <fstream>
#include <iterator>
#include <memory>
#include <stddef.h>
#include <string>
#include <thread>
#include <vector>

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/InvalidCharsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "BatchObjects/StringSequenceBatch.h"
#include "Utils/IOUtils.h"
#include "Utils/Logger.h"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using fmt::format;
using io_utils::ThrowingOfstream;
using log_utils::Logger;
using std::ios_base;
using std::make_unique;
using std::min;
using std::next;
using std::ofstream;
using std::shared_ptr;
using std::unique_ptr;

namespace sbwt_search {

template <
  class ResultsProducer,
  class IntervalProducer,
  class InvalidCharsProducer>
class ContinuousResultsPrinter {
  private:
    shared_ptr<ResultsProducer> results_producer;
    shared_ptr<IntervalProducer> interval_producer;
    shared_ptr<InvalidCharsProducer> invalid_chars_producer;
    shared_ptr<IntervalBatch> interval_batch;
    vector<string> filenames;
    size_t chars_index = 0, results_index = 0, line_index = 0;
    size_t invalid_chars_left = 0;
    size_t chars_before_newline_index = 0;

  protected:
    vector<string>::iterator current_filename;
    shared_ptr<ResultsBatch> results_batch;
    shared_ptr<InvalidCharsBatch> invalid_chars_batch;
    const uint kmer_size;
    unique_ptr<ThrowingOfstream> stream;

  public:
    ContinuousResultsPrinter(
      shared_ptr<ResultsProducer> results_producer,
      shared_ptr<IntervalProducer> interval_producer,
      shared_ptr<InvalidCharsProducer> invalid_chars_producer,
      const vector<string> &filenames,
      uint kmer_size
    ):
        results_producer(results_producer),
        interval_producer(interval_producer),
        invalid_chars_producer(invalid_chars_producer),
        filenames(filenames),
        kmer_size(kmer_size) {
      current_filename = this->filenames.begin();
    }

    auto read_and_generate() -> void {
      if (current_filename == filenames.end()) { return; }
      do_start_next_file();
      for (uint batch_idx = 0;
           (*interval_producer >> interval_batch)
           & (*invalid_chars_producer >> invalid_chars_batch)
           & (*results_producer >> results_batch);
           ++batch_idx) {
        Logger::log_timed_event(
          "ResultsPrinter",
          Logger::EVENT_STATE::START,
          format("batch {}", batch_idx)
        );
        process_batch();
        Logger::log_timed_event(
          "ResultsPrinter",
          Logger::EVENT_STATE::STOP,
          format("batch {}", batch_idx)
        );
      }
      do_at_file_end();
    }

  private:
    auto process_batch() -> void {
      chars_index = results_index = line_index = 0;
      chars_before_newline_index = 0;
      for (auto newlines_before_newfile:
           interval_batch->newlines_before_newfile) {
        process_file(newlines_before_newfile);
        if (results_index >= results_batch->results.size()) { return; }
        do_start_next_file();
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
          do_with_newline();
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
        if (invalid_chars_batch->invalid_chars[chars_index + kmer_size - 1]) {
          invalid_chars_left = kmer_size;
        }
        process_result(
          results_batch->results[results_index],
          results_batch->results[results_index] != size_t(-1),
          invalid_chars_left == 0
        );
        if (invalid_chars_left > 0) { --invalid_chars_left; }
        ++results_index;
        ++chars_index;
      }
    }

    auto get_invalid_chars_left_first_kmer() -> size_t {
      auto &invalid_chars = invalid_chars_batch->invalid_chars;
      auto limit = min({ chars_index + kmer_size,
                         invalid_chars.size(),
                         (*interval_batch->chars_before_newline
                         )[chars_before_newline_index] });
      if (limit <= chars_index) { return 0; }
      for (size_t i = limit; i > chars_index; --i) {
        if (invalid_chars[i - 1] == 1) { return i - chars_index; }
      }
      return 0;
    }

    auto process_result(size_t result, bool found, bool valid) {
      if (!valid) {
        do_invalid_result();
      } else if (!found) {
        do_not_found_result();
      } else {
        do_result(result);
      }
    }

  protected:
    virtual auto do_start_next_file() -> void {
      if (current_filename != filenames.begin()) { do_at_file_end(); }
      stream = make_unique<ThrowingOfstream>(
        *current_filename, ios_base::binary | ios_base::out
      );
      current_filename = next(current_filename);
    }

    virtual auto do_at_file_end() -> void {}
    virtual auto do_invalid_result() -> void = 0;
    virtual auto do_not_found_result() -> void = 0;
    virtual auto do_result(size_t result) -> void = 0;
    virtual auto do_with_newline() -> void = 0;
};

}  // namespace sbwt_search
#endif
