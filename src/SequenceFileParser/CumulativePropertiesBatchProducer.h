#ifndef CUMULATIVE_PROPERTIES_BATCH_PRODUCER_H
#define CUMULATIVE_PROPERTIES_BATCH_PRODUCER_H

/**
 * @file CumulativePropertiesBatchProducer.h
 * @brief Builds the CimulativeProperties Batches
 * */

#include <memory>
#include <string>

#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class CumulativePropertiesBatch;
class ContinuousSequenceFileParser;
}  // namespace sbwt_search

using design_utils::SharedBatchesProducer;
using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

class CumulativePropertiesBatchProducer:
    public SharedBatchesProducer<CumulativePropertiesBatch> {
    friend ContinuousSequenceFileParser;

  private:
    uint kmer_size;
    u64 max_strings_per_batch;

    CumulativePropertiesBatchProducer(
      u64 max_batches, u64 max_strings_per_batch, uint kmer_size
    );
    auto no_more_sequences() -> bool;
    auto get_default_value() -> shared_ptr<CumulativePropertiesBatch> override;
    auto add_string(const string &s) -> void;
    auto terminate_batch() -> void;
    auto reset_batch(shared_ptr<CumulativePropertiesBatch> &batch) -> void;
    auto start_new_batch() -> void;
    auto set_finished_reading() -> void;
    auto do_at_batch_start() -> void override;
};
}  // namespace sbwt_search

#endif
