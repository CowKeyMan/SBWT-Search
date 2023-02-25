#include <memory>

#include "SequenceFileParser/StringBreakBatchProducer.h"

using std::make_shared;

namespace sbwt_search {

StringBreakBatchProducer::StringBreakBatchProducer(const u64 max_batches):
    SharedBatchesProducer<StringBreakBatch>(max_batches) {
  initialise_batches();
}

auto StringBreakBatchProducer::get_default_value()
  -> shared_ptr<StringBreakBatch> {
  return make_shared<StringBreakBatch>();
}

auto StringBreakBatchProducer::set(
  const vector<u64> &chars_before_newline, u64 string_size
) -> void {
  current_write()->chars_before_newline = &chars_before_newline;
  current_write()->string_size = string_size;
}

}  // namespace sbwt_search
