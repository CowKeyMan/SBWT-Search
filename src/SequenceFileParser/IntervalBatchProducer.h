#ifndef INTERVAL_BATCH_PRODUCER_H
#define INTERVAL_BATCH_PRODUCER_H

/**
 * @file IntervalBatchProducer.h
 * @brief Builds the IntervalBatch
 * */

#include <memory>

#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class IntervalBatch;
class ContinuousSequenceFileParser;
}  // namespace sbwt_search

using design_utils::SharedBatchesProducer;
using std::shared_ptr;

namespace sbwt_search {

class IntervalBatchProducer: public SharedBatchesProducer<IntervalBatch> {
    friend ContinuousSequenceFileParser;

  public:
    IntervalBatchProducer(uint max_batches);

  private:
    auto add_file_end(size_t characters) -> void;
    auto do_at_batch_start() -> void override;
    auto do_at_batch_finish() -> void override;
    auto get_default_value() -> shared_ptr<IntervalBatch> override;
    auto set_string_breaks(const vector<size_t> &string_breaks) -> void;
};

}  // namespace sbwt_search

#endif
