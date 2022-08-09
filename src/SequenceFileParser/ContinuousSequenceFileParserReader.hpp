#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_READER_HPP
#define CONTINUOUS_SEQUENCE_FILE_PARSER_READER_HPP

/**
 * @file ContinuousSequenceFileParserReader.hpp
 * @brief Continuously reads sequences from a file or multiple files into a
 * buffer
 * */

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "SequenceFileParser/SequenceFileParser.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/Semaphore.hpp"
#include "Utils/TypeDefinitions.h"

using std::cerr;
using std::make_shared;
using std::queue;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;
using threading_utils::BoundedSemaphore;
using threading_utils::Semaphore;

namespace sbwt_search {

class ContinuousSequenceFileParserReader {
    const vector<string> &filenames;
    const u32 characters_per_send;
    const u64 max_characters_per_batch;
    u64 batch_size = 0, characters_not_consumed = 0;
    queue<shared_ptr<vector<string>>> &batches;
    BoundedSemaphore &batch_semaphore;
    Semaphore &character_semaphore;

  public:
    ContinuousSequenceFileParserReader(
      const vector<string> &filenames,
      queue<shared_ptr<vector<string>>> &batches,
      BoundedSemaphore &batch_semaphore,
      Semaphore &character_semaphore,
      const u32 characters_per_send,
      const u64 max_characters_per_batch
    ):
        filenames(filenames),
        batches(batches),
        batch_semaphore(batch_semaphore),
        character_semaphore(character_semaphore),
        characters_per_send(characters_per_send),
        max_characters_per_batch(max_characters_per_batch) {}

    auto read() -> void {
      start_new_batch();
      for (int i = 0; i < filenames.size(); ++i) {
        try {
          process_file(filenames[i]);
        } catch (runtime_error &e) { cerr << e.what() << '\n'; }
      }
      terminate_batch();
    }

  private:
    auto start_new_batch() -> void {
      terminate_batch();
      batches.push(make_shared<vector<string>>());
      batch_size = 0;
      characters_not_consumed = 0;
      batch_semaphore.acquire();
    }

    auto terminate_batch() -> void {
      if (characters_not_consumed > 0) { character_semaphore.release(); }
    }

    auto process_file(const string &filename) -> void {
      SequenceFileParser parser(filename);
      string s;
      u64 string_index = 0;
      while (parser >> s) { process_string(filename, s, string_index++); }
    }

    auto process_string(
      const string &filename, const string &s, const u64 string_index
    ) -> void {
      if (string_larger_than_limit(s)) {
        cerr << "The string at position " + to_string(string_index)
                  + " in file " + filename + " is too large\n";
        return;
      }
      if (!string_fits_in_batch(s)) { start_new_batch(); }
      add_string(s);
    }

    auto string_larger_than_limit(const string &s) -> bool {
      return s.size() > max_characters_per_batch;
    }

    auto string_fits_in_batch(const string &s) -> bool {
      return s.size() + batch_size <= max_characters_per_batch;
    }

    auto add_string(const string &s) -> void {
      batches.back()->push_back(s);
      batch_size += s.size();
      characters_not_consumed += s.size();
      auto sends_available = characters_not_consumed / characters_per_send;
      for (int i = 0; i < sends_available; ++i) {
        character_semaphore.release();
      }
      characters_not_consumed %= characters_per_send;
    }
};

}

#endif
