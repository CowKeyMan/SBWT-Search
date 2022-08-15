#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_H
#define CONTINUOUS_SEQUENCE_FILE_PARSER_H

/**
 * @file ContinuousSequenceFileParser.h
 * @brief Continuously reads sequences from a file or multiple files into a
 * buffer. Then it can serve these sequences to its consumers
 * */

#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "BatchObjects/CumulativePropertiesBatch.hpp"
#include "BatchObjects/StringSequenceBatch.hpp"
#include "SequenceFileParser/CumulativePropertiesBatchProducer.hpp"
#include "SequenceFileParser/SequenceFileParser.h"
#include "SequenceFileParser/StringSequenceBatchProducer.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::cerr;
using std::make_shared;
using std::queue;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;
using structure_utils::CircularBuffer;

namespace sbwt_search {

class ContinuousSequenceFileParser {
  private:
    const u64 max_chars_per_batch, max_strings_per_batch;
    u64 current_batch_size = 0, current_batch_strings = 0;
    const uint num_readers;
    const vector<string> filenames;
    StringSequenceBatchProducer string_sequence_batch_producer;
    CumulativePropertiesBatchProducer cumulative_properties_batch_producer;
    const uint bits_split = 64;

  public:
    ContinuousSequenceFileParser(
      const vector<string> &filenames,
      const uint kmer_size,
      const u64 max_chars_per_batch = 1000,
      const u64 max_strings_per_batch = 1000,
      const uint num_readers = 1,
      const u64 max_batches = 5,
      const uint bits_split = 64
    ):
        filenames(filenames),
        num_readers(num_readers),
        bits_split(bits_split),
        max_strings_per_batch(max_strings_per_batch),
        max_chars_per_batch(
          get_max_chars_per_batch(max_chars_per_batch, bits_split)
        ),
        string_sequence_batch_producer(
          max_strings_per_batch,
          get_max_chars_per_batch(max_chars_per_batch, bits_split),
          max_batches,
          num_readers,
          bits_split
        ),
        cumulative_properties_batch_producer(
          max_batches, max_strings_per_batch, kmer_size
        ) {}

    auto get_max_chars_per_batch(u64 value, uint bits_split) -> const u64 {
      auto result = round_down<u64>(value, bits_split / 2);
      if (result == 0) { result = bits_split / 2; };
      return result;
    }

    void read_and_generate() {
      start_new_batch();
      for (auto &filename: filenames) {
        try {
          process_file(filename);
        } catch (runtime_error &e) { cerr << e.what() << '\n'; }
      }
      terminate_batch();
      string_sequence_batch_producer.set_finished_reading();
      cumulative_properties_batch_producer.set_finished_reading();
    }

    bool operator>>(shared_ptr<const StringSequenceBatch> &batch) {
      return string_sequence_batch_producer >> batch;
    }

    bool operator>>(shared_ptr<const CumulativePropertiesBatch> &batch) {
      return cumulative_properties_batch_producer >> batch;
    }

  private:
    auto start_new_batch() -> void {
      string_sequence_batch_producer.start_new_batch();
      cumulative_properties_batch_producer.start_new_batch();
      current_batch_size = current_batch_strings = 0;
    }

    auto terminate_batch() -> void {
      string_sequence_batch_producer.terminate_batch();
      cumulative_properties_batch_producer.terminate_batch();
    }

    auto process_file(const string &filename) -> void {
      SequenceFileParser parser(filename);
      string s;
      u64 string_index = 0;
      while (parser >> s) { process_string(filename, s, string_index++); }
    }

    auto
    process_string(const string &filename, string &s, const u64 string_index)
      -> void {
      if (string_larger_than_limit(s)) {
        print_string_too_large(filename, string_index);
        return;
      }
      if (!string_fits_in_batch(s) || current_batch_strings >= max_strings_per_batch) {
        terminate_batch();
        start_new_batch();
      }
      add_string(s);
      ++current_batch_strings;
    }

    auto add_string(string &s) -> void {
      string_sequence_batch_producer.add_string(s);
      cumulative_properties_batch_producer.add_string(s);
      current_batch_size += s.size();
    }

    auto string_fits_in_batch(const string &s) -> bool {
      return s.size() + current_batch_size <= max_chars_per_batch;
    }

    auto string_larger_than_limit(const string &s) -> bool {
      return s.size() > max_chars_per_batch;
    }

    auto print_string_too_large(const string &filename, const uint string_index)
      -> void {
      cerr << "The string in file " + filename + " at position "
                + to_string(string_index) + " is too large\n";
    }
};

}
#endif
