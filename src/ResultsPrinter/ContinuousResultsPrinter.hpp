#ifndef CONTINUOUS_RESULTS_PRINTER_HPP
#define CONTINUOUS_RESULTS_PRINTER_HPP

/**
 * @file ContinuousResultsPrinter.hpp
 * @brief Gets results, intervals and list of invalid characters and prints
 * these out to disk based on the given data and filenames.
 * */
#include <climits>
#include <fstream>
#include <iterator>
#include <memory>
#include <stddef.h>
#include <string>
#include <vector>

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/StringSequenceBatch.h"
#include "Utils/TypeDefinitions.h"
#include "spdlog/spdlog.h"

using std::next;
using std::ofstream;
using std::shared_ptr;

namespace sbwt_search {

template <
  class ResultsProducer,
  class IntervalProducer,
  class InvalidCharsProducer>
class ContinuousResultsPrinter {
    // producers
    shared_ptr<ResultsProducer> results_producer;
    shared_ptr<IntervalProducer> interval_producer;
    shared_ptr<InvalidCharsProducer> invalid_chars_producer;
    // batch objects
    shared_ptr<vector<u64>> results;
    shared_ptr<vector<char>> invalid_chars;
    shared_ptr<IntervalBatch> interval_batch;
    // other parameters
    const uint kmer_size;
    vector<string> filenames;
    // runtime objects
    ofstream stream;
    vector<string>::iterator current_filename;
    size_t string_index = 0, char_index = 0, invalid_index = 0;

  public:
    ContinuousResultsPrinter(
      shared_ptr<ResultsProducer> results_producer,
      shared_ptr<IntervalProducer> interval_producer,
      shared_ptr<InvalidCharsProducer> invalid_chars_producer,
      vector<string> &filenames,
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
      open_next_file();
      for (uint batch_idx = 0; (*results_producer >> results)
                               & (*interval_producer >> interval_batch)
                               & (*invalid_chars_producer >> invalid_chars);
           ++batch_idx) {
        spdlog::trace("ResultsPrinter has started batch {}", batch_idx);
        process_batch();
        spdlog::trace("ResultsPrinter has finished batch {}", batch_idx);
      }
    }

  private:
    auto process_batch() -> void {
      string_index = 0, char_index = 0, invalid_index = 0;
      for (auto file_length: interval_batch->strings_before_newfile) {
        if (!stream.is_open()) { open_next_file(); }
        print_words(string_index, file_length);
        string_index += file_length;
        if (file_length != ULLONG_MAX) { stream.close(); }
      }
    }

    auto open_next_file() {
      stream.open(*current_filename, std::ios::out);
      current_filename = next(current_filename);
    }

    auto print_words(size_t string_index, size_t file_length) {
      auto total_strings = interval_batch->string_lengths.size();
      for (size_t i = string_index;
           (file_length == ULLONG_MAX || i < file_length + string_index)
           && i < total_strings;
           ++i) {
        auto string_length = interval_batch->string_lengths[i];
        auto num_chars = string_length - kmer_size + 1;
        if (string_length < kmer_size) { num_chars = 0; }
        print_word(char_index, invalid_index, num_chars, string_length);
        char_index += num_chars;
        invalid_index += string_length;
      }
    }

    auto print_word(
      size_t char_index,
      size_t invalid_index,
      size_t num_chars,
      size_t string_length
    ) -> void {
      uint invalid_chars_left
        = get_invalid_chars_left_first_kmer(invalid_index);
      for (int i = char_index; i < char_index + num_chars; ++i) {
        auto furthest_index = invalid_index + kmer_size - 1 + i - char_index;
        if (furthest_index < invalid_index + string_length && (*invalid_chars)[furthest_index]) {
          invalid_chars_left = kmer_size;
        }
        if (invalid_chars_left > 0) {
          stream << "-2";
          --invalid_chars_left;
        } else if ((*results)[i] == u64(-1)) {
          stream << "-1";
        } else {
          stream << (*results)[i];
        }
        if (i + 1 != char_index + num_chars) { stream << ' '; }
      }
      stream << '\n';
    }

    auto get_invalid_chars_left_first_kmer(size_t first_invalid_index) -> uint {
      for (uint i = kmer_size; i > 0; --i) {
        if ((*invalid_chars)[i - 1 + first_invalid_index]) { return i; }
      }
      return 0;
    }
};
}  // namespace sbwt_search
#endif
