#ifndef INTERVAL_BATCH_PRODUCER_HPP
#define INTERVAL_BATCH_PRODUCER_HPP

/**
 * @file IntervalBatchProducer.hpp
 * @brief Builds the IntervalBatch
 * */

#include <memory>
#include <string>

#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class IntervalBatch;
}  // namespace sbwt_search

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
    IntervalBatchProducer(u64 max_batches, u64 max_strings_per_batch);
    auto add_string(const string &s) -> void;
    auto file_end() -> void;
    auto terminate_batch() -> void;
    auto start_new_batch() -> void;
    bool operator>>(shared_ptr<IntervalBatch> &batch);
    auto set_finished_reading() -> void;

  private:
    auto no_more_sequences() -> bool;
    auto get_empty_batch(const u64 max_strings_per_batch)
      -> shared_ptr<IntervalBatch>;
};
}  // namespace sbwt_search

#endif
