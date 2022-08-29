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
#include "Utils/CircularBuffer.hpp"
#include "Utils/Logger.h"
#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using fmt::format;
using log_utils::Logger;
using std::make_shared;
using std::shared_ptr;
using std::vector;

using design_utils::SharedBatchesProducer;

namespace sbwt_search {

template <class CumulativePropertiesProducer>
class ContinuousPositionsBuilder: public SharedBatchesProducer<vector<u64>> {
  private:
    shared_ptr<CumulativePropertiesProducer> producer;
    PositionsBuilder builder;
    const u64 max_positions_per_batch;
    const uint kmer_size;
    shared_ptr<CumulativePropertiesBatch> read_batch;

  public:
    ContinuousPositionsBuilder(
      shared_ptr<CumulativePropertiesProducer> producer,
      const uint kmer_size,
      const u64 max_positions_per_batch = 999,
      const unsigned int max_batches = 10
    ):
        producer(producer),
        kmer_size(kmer_size),
        max_positions_per_batch(max_positions_per_batch),
        builder(kmer_size),
        SharedBatchesProducer<vector<u64>>(max_batches) {}

  protected:
    auto get_default_value() -> shared_ptr<vector<u64>> override {
      return make_shared<vector<u64>>(max_positions_per_batch);
    }

    auto continue_read_condition() -> bool override {
      return *producer >> read_batch;
    }

    auto generate() -> void override {
      builder.build_positions(
        read_batch->cumsum_positions_per_string,
        read_batch->cumsum_string_lengths,
        *batches.current_write()
      );
    }

    auto do_at_batch_start(unsigned int batch_id) -> void override {
      SharedBatchesProducer<vector<u64>>::do_at_batch_start();
      Logger::log_timed_event(
        "PositionsBuilder",
        Logger::EVENT_STATE::START,
        format("batch {}", batch_id)
      );
    }

    auto do_at_batch_finish(unsigned int batch_id) -> void override {
      Logger::log_timed_event(
        "PositionsBuilder",
        Logger::EVENT_STATE::STOP,
        format("batch {}", batch_id)
      );
      SharedBatchesProducer<vector<u64>>::do_at_batch_finish();
    }
};

}  // namespace sbwt_search

#endif
