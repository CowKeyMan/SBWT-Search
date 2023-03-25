#include <string>

#include <fmt/format.h>

#include "ColorResultsPrinter/AsciiContinuousColorResultsPrinter.h"

namespace sbwt_search {

AsciiContinuousColorResultsPrinter::AsciiContinuousColorResultsPrinter(
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
  bool include_invalid_
):
    Base(
      stream_id_,
      interval_batch_producer_,
      read_statistics_batch_producer_,
      results_batch_producer_,
      filenames_,
      num_colors_,
      threshold_,
      include_not_found_,
      include_invalid_
    ) {}

auto AsciiContinuousColorResultsPrinter::do_get_extension() -> string {
  return ".txt";
}
auto AsciiContinuousColorResultsPrinter::do_get_format() -> string {
  return "ascii";
}
auto AsciiContinuousColorResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

auto AsciiContinuousColorResultsPrinter::do_with_newline() -> u64 {
  *out_stream << '\n';
  return 1;
}
auto AsciiContinuousColorResultsPrinter::do_with_space() -> u64 {
  *out_stream << ' ';
  return 1;
}
auto AsciiContinuousColorResultsPrinter::do_with_result(u64 result) -> u64 {
  *out_stream << result;
  return std::to_string(result).size();
}

}  // namespace sbwt_search
