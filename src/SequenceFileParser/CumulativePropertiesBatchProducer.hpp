#ifndef CUMULATIVE_PROPERTIES_BATCH_PRODUCER_HPP
#define CUMULATIVE_PROPERTIES_BATCH_PRODUCER_HPP

/**
 * @file stringsequencebatchbuilder.hpp
 * @brief takes care of building and sending the stringsequencebatch
 * */

#include <memory>
#include <string>

#include "BatchObjects/CumulativePropertiesBatch.hpp"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"

using std::make_shared;
using std::shared_ptr;
using std::string;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

class CumulativePropertiesBatchProducer {
  private:
    CircularBuffer<shared_ptr<CumulativePropertiesBatch>> batches;
    const uint max_strings_per_batch;
    bool finished_reading = false;
    BoundedSemaphore semaphore;
    uint kmer_size;

  public:
    CumulativePropertiesBatchProducer(
      u64 max_batches, u64 max_strings_per_batch, uint kmer_size
    ):
        batches(max_batches + 1),
        max_strings_per_batch(max_strings_per_batch),
        semaphore(0, max_batches),
        kmer_size(kmer_size) {
      for (int i = 0; i < batches.size(); ++i) {
        batches.set(i, get_empty_cumsum_batch());
      }
    }

  private:
    auto get_empty_cumsum_batch() -> shared_ptr<CumulativePropertiesBatch> {
      auto batch = make_shared<CumulativePropertiesBatch>();
      batch->cumsum_positions_per_string.reserve(max_strings_per_batch);
      batch->cumsum_positions_per_string.push_back(0);
      batch->cumsum_string_lengths.reserve(max_strings_per_batch);
      batch->cumsum_string_lengths.push_back(0);
      return batch;
    }

  public:
    auto add_string(const string &s) -> void {
      auto &batch = batches.current_write();
      auto new_positions = 0;
      if (s.size() > kmer_size) { new_positions = s.size() - kmer_size + 1; }
      batch->cumsum_positions_per_string.push_back(
        batch->cumsum_positions_per_string.back() + new_positions
      );
      batch->cumsum_string_lengths.push_back(
        batch->cumsum_string_lengths.back() + s.size()
      );
    }

    auto terminate_batch() -> void {
      batches.step_write();
      semaphore.release();
    }

    auto start_new_batch() -> void {
      reset_batch(batches.current_write());
    }

  private:
    auto reset_batch(shared_ptr<CumulativePropertiesBatch> &batch) -> void {
      batch->cumsum_positions_per_string.resize(1);
      batch->cumsum_string_lengths.resize(1);
    }

  public:
    bool operator>>(shared_ptr<const CumulativePropertiesBatch> &batch) {
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
