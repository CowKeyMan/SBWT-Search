#ifndef STRING_BREAK_BATCH_PRODUCER_H
#define STRING_BREAK_BATCH_PRODUCER_H

/**
 * @file StringBreakBatchProducer.h
 * @brief In charge of storing and handing out the locations where one sequence
 * ends and another begins, stored in the StringBreakBatch
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

  explicit StringBreakBatchProducer(u64 max_batches);

  ~StringBreakBatchProducer() override = default;

private:
  auto get_default_value() -> shared_ptr<StringBreakBatch> override;
  auto set(const vector<u64> &chars_before_newline, u64 string_size) -> void;
};

}  // namespace sbwt_search

#endif
