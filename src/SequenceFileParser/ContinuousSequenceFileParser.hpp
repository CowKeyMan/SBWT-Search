#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_H
#define CONTINUOUS_SEQUENCE_FILE_PARSER_H

/**
 * @file ContinuousSequenceFileParser.h
 * @brief Continuously reads sequences from a file or multiple files into a
 * buffer. Then it can serve these sequences to its consumers
 * */

#include <climits>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "SequenceFileParser/SequenceFileParser.h"
#include "SequenceFileParser/StringSequenceBatch.hpp"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/MathUtils.hpp"
#include "Utils/TypeDefinitions.h"

using std::cerr;
using std::move;
using std::queue;
using std::runtime_error;
using std::string;
using std::to_string;
using std::vector;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

class ContinuousSequenceFileParser {
  private:
    queue<unique_ptr<StringSequenceBatch>> sequence_batches;
    unique_ptr<StringSequenceBatch> sequence_batch;
    const uint bits_split;
    u64 sequence_batch_size, characters_to_next_index, characters_per_reader,
      max_characters_per_batch;
    const uint readers_amount;
    BoundedSemaphore batch_semaphore;
    const vector<string> filenames;

    bool finished_reading = false;

  public:
    ContinuousSequenceFileParser(
      const vector<string> &filenames,
      const u64 max_characters_per_batch = UINT_MAX,
      const uint readers_amount = 1,
      const u64 max_batches = UINT_MAX
    ):
        filenames(filenames),
        bits_split(64),
        readers_amount(readers_amount),
        batch_semaphore(0, max_batches),
        max_characters_per_batch(
          round_down<u64>(max_characters_per_batch, bits_split / 2)
        ) {
      if (this->max_characters_per_batch == 0) {
        this->max_characters_per_batch = bits_split / 2;
      }
      this->characters_per_reader = round_up<u64>(
        this->max_characters_per_batch / readers_amount, bits_split / 2
      );
    }

  public:
    void read_and_generate() {
      start_new_batch();
      for (auto &filename: filenames) {
        try {
          process_file(filename);
        } catch (runtime_error &e) { cerr << e.what() << '\n'; }
      }
      terminate_batch();
      finished_reading = true;
    }

  private:
    auto start_new_batch() -> void {
      initialise_sequence_batch();
      sequence_batch_size = 0;
      characters_to_next_index = characters_per_reader;
    }

    auto initialise_sequence_batch() -> void {
      sequence_batch = make_unique<StringSequenceBatch>();
      sequence_batch->character_indexes.reserve(readers_amount + 1);
      sequence_batch->string_indexes.reserve(readers_amount + 1);
      sequence_batch->character_indexes.push_back(0);
      sequence_batch->string_indexes.push_back(0);
    }

    auto process_file(const string &filename) -> void {
      SequenceFileParser parser(filename);
      string s;
      u64 string_index = 0;
      while (parser >> s) { process_string(filename, s, string_index++); }
    }

    auto terminate_batch() -> void {
      for (uint i = sequence_batch->string_indexes.size();
           i < readers_amount + 1;
           ++i) {
        sequence_batch->string_indexes.push_back(sequence_batch->buffer.size());
        sequence_batch->character_indexes.push_back(0);
      }
      sequence_batches.push(move(sequence_batch));
      batch_semaphore.release();
    }

    auto process_string(
      const string &filename, const string &s, const u64 string_index
    ) -> void {
      if (string_larger_than_limit(s)) {
        print_string_too_large(filename, string_index);
        return;
      }
      if (!string_fits_in_batch(s)) {
        terminate_batch();
        start_new_batch();
      }
      add_string(s);
    }

    auto string_larger_than_limit(const string &s) -> bool {
      return s.size() > max_characters_per_batch;
    }

    auto print_string_too_large(const string &filename, const uint string_index)
      -> void {
      cerr << "The string in file " + filename + " at position "
                + to_string(string_index) + " is too large\n";
    }

    auto string_fits_in_batch(const string &s) -> bool {
      return s.size() + sequence_batch_size <= max_characters_per_batch;
    }

    auto add_string(const string &s) -> void {
      if (s.size() > characters_to_next_index) {
        sequence_batch->character_indexes.push_back(characters_to_next_index);
        sequence_batch->string_indexes.push_back(sequence_batch->buffer.size());
        characters_to_next_index
          = characters_per_reader - (s.size() - characters_to_next_index);
      } else {
        characters_to_next_index -= s.size();
      }
      sequence_batch_size += s.size();
      sequence_batch->buffer.push_back(move(s));
    }

  public:
    bool operator>>(unique_ptr<StringSequenceBatch> &sb) {
      if (no_more_sequences()) { return false; }
      batch_semaphore.acquire();
      sb = move(sequence_batches.front());
      sequence_batches.pop();
      return true;
    }

  private:
    auto no_more_sequences() -> bool {
      return finished_reading && sequence_batches.empty();
    }
};

}
#endif
