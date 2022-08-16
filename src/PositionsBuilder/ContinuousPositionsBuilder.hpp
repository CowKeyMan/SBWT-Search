#ifndef CONTINUOUS_POSITIONS_BUILDER_H
#define CONTINUOUS_POSITIONS_BUILDER_H

/**
 * @file ContinuousPositionsBuilder.hpp
 * @brief Builds the positions of the valid bit sequences in a buffer and then
 * passes them on
 * */

#include <memory>
#include <vector>

#include "BatchObjects/CumulativePropertiesBatch.hpp"
#include "PositionsBuilder/PositionsBuilder.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/TypeDefinitions.h"

using std::shared_ptr;
using std::vector;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

template <class CumulativePropertiesProducer>
class ContinuousPositionsBuilder {
    shared_ptr<CumulativePropertiesProducer> producer;
    CircularBuffer<vector<u64>> batches;
    BoundedSemaphore batch_semaphore;
    const u64 max_positions_per_batch;
    bool finished = false;
    PositionsBuilder builder;
    const uint kmer_size;

  public:
    ContinuousPositionsBuilder(
      shared_ptr<CumulativePropertiesProducer> producer,
      const uint kmer_size,
      const u64 max_positions_per_batch = 999,
      const u64 max_batches = 10
    ):
        producer(producer),
        kmer_size(kmer_size),
        max_positions_per_batch(max_positions_per_batch),
        batch_semaphore(0, max_batches),
        batches(max_batches + 1, vector<u64>(max_positions_per_batch)),
        builder(kmer_size) {}

    auto read_and_generate() -> void {
      shared_ptr<CumulativePropertiesBatch> read_batch;
      while (*producer >> read_batch) {
        builder.build_positions(
          read_batch->cumsum_positions_per_string,
          read_batch->cumsum_string_lengths,
          batches.current_write()
        );
        batches.step_write();
        batch_semaphore.release();
      }
      finished = true;
      batch_semaphore.release();
    }

    bool operator>>(vector<u64> &batch) {
      batch_semaphore.acquire();
      if (finished && batches.empty()) { return false; }
      batch = batches.current_read();
      batches.step_read();
      return true;
    }
};

}

#endif
