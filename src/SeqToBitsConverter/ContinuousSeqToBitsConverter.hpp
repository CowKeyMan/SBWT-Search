#ifndef CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP
#define CONTINUOUS_SEQ_TO_BITS_CONVERTER_HPP

/**
 * @file ContinuousSeqToBitsConverter.hpp
 * @brief Class for converting character sequences continuously, with parallel
 * capabilities
 * */

#include <iterator>
#include <list>
#include <memory>
#include <queue>
#include <tuple>

#include "SeqToBitsConverter/CharToBits.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/ObserverPattern.hpp"
#include "Utils/TypeDefinitions.h"

using design_utils::Subject;
using std::list;
using std::make_tuple;
using std::next;
using std::queue;
using std::shared_ptr;
using std::string;
using std::tie;
using std::tuple;
using std::vector;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

template <class ContinuousSequenceParser>
class ContinuousSeqToBitsConverter: Subject<shared_ptr<vector<string>>> {
  private:
    ContinuousSequenceParser &parser;
    BoundedSemaphore write_semaphore;
    list<vector<u64>> write_batches;
    list<tuple<shared_ptr<vector<string>>, uint>> read_batches;
    uint threads_total;
    CharToBits char_to_bits;
    u64 max_characters_per_batch;

  public:
    ContinuousSeqToBitsConverter(
      ContinuousSequenceParser &parser,
      u64 max_batches,
      uint threads_total,
      u64 max_characters_per_batch
    ):
        parser(parser),
        write_semaphore(0, max_batches * 4),
        threads_total(threads_total),
        max_characters_per_batch(max_characters_per_batch) {}

  public:
    void generate() {
      static auto current_read_batch = read_batches.begin();
      static auto current_write_batch = write_batches.begin();
      static u64 string_index, character_index;
      static u32 batch_index, in_batch_index, previous_batch_index = 0;
#pragma omp threadprivate( \
  current_read_batch,      \
  current_write_batch,     \
  batch_index,             \
  in_batch_index,          \
  string_index,            \
  character_index,         \
  previous_batch_index     \
)
      while (parser
             >> tie(batch_index, in_batch_index, string_index, character_index)
      ) {
        while (batch_index != previous_batch_index) {
          current_read_batch = next(current_read_batch);
          ++previous_batch_index;
#pragma omp critical
          {
            auto &[_, counter] = *current_read_batch;
            --counter;
            if (counter == 0) { read_batches.pop_front(); }
          }
          current_write_batch = next(current_write_batch);
          write_semaphore.release();
        }
        auto &[batch, _] = *current_read_batch;
        (*current_write_batch)[in_batch_index]
          = convert_int(*batch, string_index, character_index);
      }
      write_semaphore.release();
    }

    u64
    convert_int(vector<string> &batch, u64 string_index, u64 character_index) {
      u64 result = 0;
      for (u64 internal_shift = 62; internal_shift < 64; internal_shift -= 2) {
        if (end_of_string(batch, string_index, character_index)) {
          ++string_index;
          character_index = 0;
        }
        if (string_index == batch.size()) { return result; }
        char c = batch[string_index][character_index++];
        result |= (char_to_bits(c) << internal_shift);
      }
      return result;
    }

    bool end_of_string(
      vector<string> &batch, u64 string_index, u64 character_index
    ) {
      return character_index == batch[string_index].size();
    }

    void update(shared_ptr<vector<string>> buffer) {
      read_batches.push_back(make_tuple(buffer, threads_total));
      write_batches.push_back(vector<u64>(max_characters_per_batch));
    }

    bool operator>>(vector<u64> &batch) {
      for (int i = 0; i < threads_total; ++i) { write_semaphore.acquire(); }
      if (write_batches.size() == 0) { return false; }
      batch = move(write_batches.front());
      write_batches.pop_front();
      return true;
    }
};
}

#endif
