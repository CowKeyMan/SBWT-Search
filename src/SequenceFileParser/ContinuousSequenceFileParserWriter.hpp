#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_PRODUCER_HPP
#define CONTINUOUS_SEQUENCE_FILE_PARSER_PRODUCER_HPP

/**
 * @file ContinuousSequenceFileParserWriter.hpp
 * @brief Gives the next character positions to be processed from the buffer
 * */

#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "Utils/BoundedSemaphore.hpp"
#include "Utils/Semaphore.hpp"
#include "Utils/TypeDefinitions.h"

using std::queue;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::tuple;
using std::vector;
using threading_utils::BoundedSemaphore;
using threading_utils::Semaphore;

namespace sbwt_search {

class ContinuousSequenceFileParserWriter {
    const uint characters_per_send;
    queue<shared_ptr<vector<string>>> &batches;
    BoundedSemaphore &batch_semaphore;
    Semaphore &character_semaphore;
    u64 string_index = 0, character_index = 0;
    u32 batch_index = 0;
    bool &finished_reading;
    bool first_read = true;

  public:
    ContinuousSequenceFileParserWriter(
      queue<shared_ptr<vector<string>>> &batches,
      BoundedSemaphore &batch_semaphore,
      Semaphore &character_semaphore,
      const u32 characters_per_send,
      bool &finished_reading
    ):
        batches(batches),
        batch_semaphore(batch_semaphore),
        character_semaphore(character_semaphore),
        characters_per_send(characters_per_send),
        finished_reading(finished_reading) {}

    auto operator>>(tuple<u32&, u64&, u64&>& t) -> bool {
      character_semaphore.acquire();
#pragma omp critical
      {
        if (first_read) {
          first_read = false;
        } else {
          recalculate_indexes();
        }
        if (!is_over()) { set_output_variables(t); }
      }
      return !is_over();
    }

  private:
    auto is_over() -> bool { return finished_reading && batches.empty(); }

    auto recalculate_indexes() -> void {
      auto characters_to_advance = characters_per_send;
      for (int i = string_index; i < batches.front()->size(); ++i) {
        const auto &s = (*batches.front())[i];
        if (target_within_string(characters_to_advance, s)) {
          character_index += characters_to_advance;
          string_index = i;
          return;
        }
        advance_to_end_of_string(characters_to_advance, s);
      }
      start_new_batch();
    }

    auto target_within_string(const u64 characters_to_advance, const string &s)
      -> bool {
      return character_index + characters_to_advance < s.size();
    }

    auto advance_to_end_of_string(uint &characters_to_advance, const string &s)
      -> void {
      auto advanced = s.size() - character_index;
      characters_to_advance -= advanced;
      character_index = 0;
    }

    auto start_new_batch() -> void {
      string_index = character_index = 0;
      batches.pop();
      batch_index++;
      batch_semaphore.release();
    }

    auto set_output_variables(tuple<u32&, u64&, u64&>& t) -> void {
      auto [batch_index, string_index, character_index] = t;
      batch_index = this->batch_index;
      string_index = this->string_index;
      character_index = this->character_index;
    }
};

}

#endif
