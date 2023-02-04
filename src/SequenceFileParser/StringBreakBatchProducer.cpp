#include <memory>

#include "SequenceFileParser/StringBreakBatchProducer.h"

using std::make_shared;

namespace sbwt_search {

StringBreakBatchProducer::StringBreakBatchProducer(const uint max_batches):
    SharedBatchesProducer<StringBreakBatch>(max_batches) {
  initialise_batches();
}

auto StringBreakBatchProducer::get_default_value()
  -> shared_ptr<StringBreakBatch> {
  return make_shared<StringBreakBatch>();
}

auto StringBreakBatchProducer::set(
  const vector<size_t> &chars_before_newline, size_t string_size
) -> void {
  get_batches().current_write()->chars_before_newline = &chars_before_newline;
  get_batches().current_write()->string_size = string_size;
}

}  // namespace sbwt_search
