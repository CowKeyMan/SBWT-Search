#include <iostream>
using namespace std;

#ifndef CONTINUOUS_RESULTS_PRINTER_HPP
#define CONTINUOUS_RESULTS_PRINTER_HPP

#include <climits>
#include <fstream>
#include <iterator>
#include <memory>

#include "BatchObjects/CumulativePropertiesBatch.hpp"
#include "BatchObjects/IntervalBatch.hpp"
#include "BatchObjects/StringSequenceBatch.hpp"
#include "Utils/TypeDefinitions.h"

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
    ResultsProducer results_producer;
    IntervalProducer interval_producer;
    InvalidCharsProducer invalid_chars_producer;
    // batch objects
    shared_ptr<vector<u64>> results, invalid_chars;
    shared_ptr<IntervalBatch> interval_batch;
    // other parameters
    const uint kmer_size;
    vector<string> filenames;
    // runtime objects
    ofstream stream;
    vector<string>::iterator current_filename;
    size_t string_index = 0;
    size_t char_index = 0;

  public:
    ContinuousResultsPrinter(
      ResultsProducer results_producer,
      InvalidCharsProducer invalid_chars_producer,
      IntervalProducer interval_producer,
      vector<string> filenames,
      uint kmer_size
    ):
        results_producer(results_producer),
        interval_producer(interval_producer),
        invalid_chars_producer(invalid_chars_producer),
        filenames(filenames),
        kmer_size(kmer_size) {
      current_filename = this->filenames.begin();
    }

    auto get_and_print() -> void {
      if (current_filename == filenames.end()) { return; }
      open_next_file();
      while ((results_producer >> results)
             & (interval_producer >> interval_batch)
             & (invalid_chars_producer >> invalid_chars)) {
        process_batch();
      }
    }

  private:
    auto process_batch() -> void {
      for (auto file_length: interval_batch->strings_before_newfile) {
        if (!stream.is_open()) {
        open_next_file();
        }
        print_words(string_index, file_length);
        string_index += file_length;
        if (file_length != ULLONG_MAX) {
        stream.close();
        }
      }
    }

    auto open_next_file() {
      stream.open(*current_filename, std::ios::out);
      current_filename = next(current_filename);
    }

    auto print_words(size_t string_index, size_t file_length) {
      auto total_strings = interval_batch->string_lengths.size();
      for (size_t i = string_index; i < file_length + string_index && i < total_strings; ++i) {
        auto string_length = interval_batch->string_lengths[i];
        if (string_length < kmer_size) {
          string_length = 0;
        } else {
          string_length = string_length - kmer_size + 1;
        }
        print_word(char_index, string_length);
        char_index += string_length;
      }
    }

    auto print_word(size_t index, size_t string_length) {
      for (int i = index; i < index + string_length; ++i) {
        if ((*invalid_chars)[i]) {
          stream << "-2";
        } else if ((*results)[i] == u64(-1)) {
          stream << "-1";
        } else {
          stream << (*results)[i];
        }
        if (index + 1 != string_length) { stream << ' '; }
      }
      stream << "\n";
    }
};
}
#endif
