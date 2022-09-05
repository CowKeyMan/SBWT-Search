#ifndef INTERVAL_BATCH_PRODUCER_H
#define INTERVAL_BATCH_PRODUCER_H

/**
 * @file IntervalBatchProducer.h
 * @brief Builds the IntervalBatch
 * */

#include <memory>
#include <string>

#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class IntervalBatch;
class ContinuousSequenceFileParser;
}  // namespace sbwt_search

using design_utils::SharedBatchesProducer;
using std::make_shared;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

class IntervalBatchProducer: public SharedBatchesProducer<IntervalBatch> {
    friend ContinuousSequenceFileParser;

  private:
    uint string_counter = 0;
    u64 max_strings_per_batch;

    IntervalBatchProducer(u64 max_batches, u64 max_strings_per_batch);
    auto add_string(const string &s) -> void;
    auto file_end() -> void;
    auto do_at_batch_start() -> void override;
    auto do_at_batch_finish() -> void override;
    auto start_new_batch() -> void;
    auto set_finished_reading() -> void;
    auto no_more_sequences() -> bool;
    auto get_default_value() -> shared_ptr<IntervalBatch> override;
};
}  // namespace sbwt_search

#endif
