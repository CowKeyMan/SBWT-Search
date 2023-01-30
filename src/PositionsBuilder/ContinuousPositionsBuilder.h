#ifndef CONTINUOUS_POSITIONS_BUILDER_H
#define CONTINUOUS_POSITIONS_BUILDER_H

/**
 * @file ContinuousPositionsBuilder.h
 * @brief Builds the positions of the valid bit sequences in a buffer and then
 * passes them on
 */

#include <memory>

#include "BatchObjects/PositionsBatch.h"
#include "BatchObjects/StringBreakBatch.h"
#include "PositionsBuilder/PositionsBuilder.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

class ContinuousPositionsBuilder: public SharedBatchesProducer<PositionsBatch> {
private:
  shared_ptr<SharedBatchesProducer<StringBreakBatch>> producer;
  shared_ptr<StringBreakBatch> read_batch;
  PositionsBuilder builder;
  size_t max_chars_per_batch;

public:
  ContinuousPositionsBuilder(
    shared_ptr<SharedBatchesProducer<StringBreakBatch>> _producer,
    uint kmer_size,
    size_t _max_chars_per_batch,
    uint max_batches
  );

protected:
  auto get_default_value() -> shared_ptr<PositionsBatch> override;
  auto continue_read_condition() -> bool override;
  auto generate() -> void override;
  auto do_at_batch_start() -> void override;
  auto do_at_batch_finish() -> void override;
};

}  // namespace sbwt_search

#endif
