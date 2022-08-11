#ifndef CONTINUOUS_SEQUENCE_FILE_PARSER_H
#define CONTINUOUS_SEQUENCE_FILE_PARSER_H

/**
 * @file ContinuousSequenceFileParser.h
 * @brief Continuously reads sequences from a file or multiple files into a
 * buffer. Then it can serve these sequences to its consumers
 * */

#include <climits>
#include <memory>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include "SequenceFileParser/ContinuousSequenceFileParserReader.hpp"
#include "SequenceFileParser/ContinuousSequenceFileParserWriter.hpp"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/ObserverPattern.hpp"
#include "Utils/Semaphore.hpp"
#include "Utils/TypeDefinitions.h"

using design_utils::Observer;
using std::make_shared;
using std::queue;
using std::shared_ptr;
using std::string;
using std::tuple;
using std::vector;
using threading_utils::BoundedSemaphore;
using threading_utils::Semaphore;

namespace sbwt_search {

class ContinuousSequenceFileParser {
  private:
    ContinuousSequenceFileParserReader reader;
    ContinuousSequenceFileParserWriter writer;
    const u32 characters_per_send;
    u64 kmer_size, max_characters_per_batch;
    uint readers_amount;
    void add_sequence(const string seq);
    queue<shared_ptr<vector<string>>> batches;
    BoundedSemaphore batch_semaphore;
    Semaphore character_semaphore;
    bool finished_reading = false;

  public:
    ContinuousSequenceFileParser(
      const vector<string> &filenames,
      const u64 kmer_size = 30,
      const u64 max_characters_per_batch = UINT_MAX,
      const u32 characters_per_send = 32,
      const uint readers_amount = 1,
      const u64 max_batches = UINT_MAX
    ):
        kmer_size(kmer_size),
        max_characters_per_batch(max_characters_per_batch),
        readers_amount(readers_amount),
        characters_per_send(characters_per_send),
        batch_semaphore(0, max_batches),
        character_semaphore(0),
        reader(
          filenames,
          batches,
          batch_semaphore,
          character_semaphore,
          characters_per_send,
          max_characters_per_batch
        ),
        writer(
          batches,
          batch_semaphore,
          character_semaphore,
          characters_per_send,
          finished_reading
        ){};

    void subscribe(Observer<shared_ptr<vector<string>>> *observer) {
      reader.subscribe(observer);
    }

    void unsubscribe(Observer<shared_ptr<vector<string>>> *observer) {
      reader.unsubscribe(observer);
    }

    void read_and_generate() {
      reader.read();
      finished_reading = true;
      free_all_consumers();
    }

  private:
    void free_all_consumers() {
      for (int i = 0; i < readers_amount; ++i) {
        character_semaphore.release();
      }
    }

  public:
    bool operator>>(tuple<u32 &, u32 &, u64 &, u64 &> t) { return writer >> t; }
};
}
#endif
