#ifndef STRING_BREAK_BATCH_PRODUCER_H
#define STRING_BREAK_BATCH_PRODUCER_H

/**
 * @file StringBreakBatchProducer.h
 * @brief In charge of storing and handing out the locations where one string
 * ends and another begins
 */

#include <algorithm>
#include <vector>

#include "BatchObjects/StringBreakBatch.h"
#include "Utils/SharedBatchesProducer.hpp"

using design_utils::SharedBatchesProducer;
using std::vector;

namespace sbwt_search {
class ContinuousSequenceFileParser;
}  // namespace sbwt_search

namespace sbwt_search {

class StringBreakBatchProducer: public SharedBatchesProducer<StringBreakBatch> {
    friend ContinuousSequenceFileParser;

  public:
    StringBreakBatchProducer(const uint max_batches);

  private:
    auto get_default_value() -> shared_ptr<StringBreakBatch> override;
    auto set(const vector<size_t> &string_breaks, size_t string_size) -> void;
};

}  // namespace sbwt_search

#endif
