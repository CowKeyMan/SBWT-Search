#include <string>

#include "ColorResultsPrinter/BinaryContinuousColorResultsPrinter.h"

namespace sbwt_search {

BinaryContinuousColorResultsPrinter::BinaryContinuousColorResultsPrinter(
  u64 stream_id_,
  shared_ptr<SharedBatchesProducer<ColorsIntervalBatch>>
    interval_batch_producer_,
  shared_ptr<SharedBatchesProducer<ReadStatisticsBatch>>
    read_statistics_batch_producer_,
  shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
    results_batch_producer_,
  const vector<string> &filenames_,
  u64 num_colors_,
  double threshold_,
  bool include_not_found_,
  bool include_invalid_,
  u64 threads,
  u64 max_reads_in_buffer,
  bool write_headers
):
    Base(
      stream_id_,
      std::move(interval_batch_producer_),
      std::move(read_statistics_batch_producer_),
      std::move(results_batch_producer_),
      filenames_,
      num_colors_,
      threshold_,
      include_not_found_,
      include_invalid_,
      threads,
      1,
      1,
      max_reads_in_buffer,
      write_headers
    ) {}

auto BinaryContinuousColorResultsPrinter::do_get_extension() -> string {
  return ".bin";
}
auto BinaryContinuousColorResultsPrinter::do_get_format() -> string {
  return "binary";
}
auto BinaryContinuousColorResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

auto BinaryContinuousColorResultsPrinter::do_with_newline(
  vector<u64>::iterator buffer
) -> u64 {
  *buffer = static_cast<u64>(-1);
  return 1;
}
auto BinaryContinuousColorResultsPrinter::do_with_result(
  vector<u64>::iterator buffer, u64 result
) -> u64 {
  *buffer = result;
  return 1;
}

}  // namespace sbwt_search
