#ifndef INTERVAL_BATCH_PRODUCER_HPP
#define INTERVAL_BATCH_PRODUCER_HPP

/**
 * @file IntervalBatchProducer.hpp
 * @brief Builds the IntervalBatch
 * */

#include <memory>
#include <string>

#include "BatchObjects/IntervalBatch.hpp"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"

using std::make_shared;
using std::shared_ptr;
using std::string;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

class IntervalBatchProducer {
  private:
    CircularBuffer<shared_ptr<IntervalBatch>> batches;
    bool finished_reading = false;
    BoundedSemaphore semaphore;
    uint string_counter = 0;

  public:
    IntervalBatchProducer(u64 max_batches, u64 max_strings_per_batch):
        batches(max_batches + 1), semaphore(0, max_batches) {
      for (int i = 0; i < batches.size(); ++i) {
        batches.set(i, get_empty_batch(max_strings_per_batch));
      }
    }

  private:
    auto get_empty_batch(const u64 max_strings_per_batch)
      -> shared_ptr<IntervalBatch> {
      auto batch = make_shared<IntervalBatch>();
      batch->string_lengths.reserve(max_strings_per_batch);
      batch->strings_before_newfile.reserve(max_strings_per_batch);
      return batch;
    }

  public:
    auto add_string(const string &s) -> void {
      batches.current_write()->string_lengths.push_back(s.size());
      string_counter++;
    }

    auto file_end() -> void {
      batches.current_write()->strings_before_newfile.push_back(string_counter);
      string_counter = 0;
    }

    auto terminate_batch() -> void {
      batches.step_write();
      semaphore.release();
    }

    auto start_new_batch() -> void {
      batches.current_write()->string_lengths.resize(0);
      batches.current_write()->strings_before_newfile.resize(0);
      string_counter = 0;
    }

  public:
    bool operator>>(shared_ptr<const IntervalBatch> &batch) {
      semaphore.acquire();
      if (no_more_sequences()) { return false; }
      batch = batches.current_read();
      batches.step_read();
      return true;
    }

    auto set_finished_reading() -> void {
      finished_reading = true;
      semaphore.release();
    }

  private:
    auto no_more_sequences() -> bool {
      return finished_reading && batches.empty();
    }
};
}

#endif
