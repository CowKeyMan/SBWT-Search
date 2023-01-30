#ifndef STRING_BREAK_BATCH_PRODUCER_H
#define STRING_BREAK_BATCH_PRODUCER_H

/**
 * @file StringBreakBatchProducer.h
 * @brief In charge of storing and handing out the locations where one string
 * ends and another begins
 */

#include <algorithm>
#include <memory>
#include <vector>

#include "BatchObjects/StringBreakBatch.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

class ContinuousSequenceFileParser;

using design_utils::SharedBatchesProducer;
using std::shared_ptr;
using std::vector;

class StringBreakBatchProducer: public SharedBatchesProducer<StringBreakBatch> {
  friend ContinuousSequenceFileParser;

public:
  StringBreakBatchProducer(StringBreakBatchProducer &) = delete;
  StringBreakBatchProducer(StringBreakBatchProducer &&) = delete;
  auto operator=(StringBreakBatchProducer &) = delete;
  auto operator=(StringBreakBatchProducer &&) = delete;

  explicit StringBreakBatchProducer(uint max_batches);

  ~StringBreakBatchProducer() override = default;

private:
  auto get_default_value() -> shared_ptr<StringBreakBatch> override;
  auto set(const vector<size_t> &chars_before_newline, size_t string_size)
    -> void;
};

}  // namespace sbwt_search

#endif
