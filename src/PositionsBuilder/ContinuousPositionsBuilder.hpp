#ifndef CONTINUOUS_POSITIONS_BUILDER_HPP
#define CONTINUOUS_POSITIONS_BUILDER_HPP

/**
 * @file ContinuousPositionsBuilder.hpp
 * @brief Builds the positions of the valid bit sequences in a buffer and then
 * passes them on
 * */

#include <memory>

#include "BatchObjects/StringBreakBatch.h"
#include "BatchObjects/PositionsBatch.h"
#include "PositionsBuilder/PositionsBuilder.h"
#include "Utils/Logger.h"
#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using fmt::format;
using log_utils::Logger;
using std::make_shared;
using std::shared_ptr;

using design_utils::SharedBatchesProducer;

namespace sbwt_search {

template <class StringBreakBatchProducer>
class ContinuousPositionsBuilder: public SharedBatchesProducer<PositionsBatch> {
  private:
    shared_ptr<StringBreakBatchProducer> producer;
    shared_ptr<StringBreakBatch> read_batch;
    PositionsBuilder builder;
    const size_t max_chars_per_batch;

  public:
    ContinuousPositionsBuilder(
      shared_ptr<StringBreakBatchProducer> _producer,
      const uint kmer_size,
      const size_t _max_chars_per_batch,
      const uint max_batches
    ):
        producer(_producer),
        max_chars_per_batch(_max_chars_per_batch),
        builder(kmer_size),
        SharedBatchesProducer<PositionsBatch>(max_batches) {
      initialise_batches();
    }

  protected:
    auto get_default_value() -> shared_ptr<PositionsBatch> override {
      auto batch = make_shared<PositionsBatch>();
      batch->positions.resize(max_chars_per_batch);
      return batch;
    }

    auto continue_read_condition() -> bool override {
      return (*producer) >> read_batch;
    }

    auto generate() -> void override {
      builder.build_positions(
        *read_batch->chars_before_newline,
        read_batch->string_size,
        batches.current_write()->positions
      );
    }

    auto do_at_batch_start() -> void override {
      SharedBatchesProducer<PositionsBatch>::do_at_batch_start();
      Logger::log_timed_event(
        "PositionsBuilder",
        Logger::EVENT_STATE::START,
        format("batch {}", batch_id)
      );
    }

    auto do_at_batch_finish() -> void override {
      Logger::log_timed_event(
        "PositionsBuilder",
        Logger::EVENT_STATE::STOP,
        format("batch {}", batch_id)
      );
      SharedBatchesProducer<PositionsBatch>::do_at_batch_finish();
    }
};

}  // namespace sbwt_search

#endif
