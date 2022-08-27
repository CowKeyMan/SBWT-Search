#ifndef CONTINUOUS_POSITIONS_BUILDER_HPP
#define CONTINUOUS_POSITIONS_BUILDER_HPP

/**
 * @file ContinuousPositionsBuilder.hpp
 * @brief Builds the positions of the valid bit sequences in a buffer and then
 * passes them on
 * */

#include <memory>
#include <vector>

#include "BatchObjects/CumulativePropertiesBatch.h"
#include "PositionsBuilder/PositionsBuilder.h"
#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/Logger.h"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using fmt::format;
using log_utils::Logger;
using std::make_shared;
using std::shared_ptr;
using std::vector;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

template <class CumulativePropertiesProducer>
class ContinuousPositionsBuilder {
    shared_ptr<CumulativePropertiesProducer> producer;
    CircularBuffer<shared_ptr<vector<u64>>> batches;
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
        batches(max_batches + 1),
        builder(kmer_size) {
      for (uint i = 0; i < batches.size(); ++i) {
        batches.set(i, make_shared<vector<u64>>(max_positions_per_batch));
      }
    }

    auto read_and_generate() -> void {
      shared_ptr<CumulativePropertiesBatch> read_batch;
      for (uint batch_idx = 0; *producer >> read_batch; ++batch_idx) {
        Logger::log_timed_event(
          "PositionsBuilder",
          Logger::EVENT_STATE::START,
          format("batch {}", batch_idx)
        );
        builder.build_positions(
          read_batch->cumsum_positions_per_string,
          read_batch->cumsum_string_lengths,
          *batches.current_write()
        );
        Logger::log_timed_event(
          "PositionsBuilder",
          Logger::EVENT_STATE::STOP,
          format("batch {}", batch_idx)
        );
        batches.step_write();
        batch_semaphore.release();
      }
      finished = true;
      batch_semaphore.release();
    }

    auto operator>>(shared_ptr<vector<u64>> &batch) -> bool {
      batch_semaphore.acquire();
      if (finished && batches.empty()) { return false; }
      batch = batches.current_read();
      batches.step_read();
      return true;
    }
};

}  // namespace sbwt_search

#endif
