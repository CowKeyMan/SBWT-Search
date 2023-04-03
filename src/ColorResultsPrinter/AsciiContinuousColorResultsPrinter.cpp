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
      max_chars_in_u64 + 1,
      1,
      max_reads_in_buffer,
      write_headers
    ),
    tiny_buffers(threads) {
  for (u64 i = 0; i < threads; ++i) {
    tiny_buffers[i].resize(max_chars_in_u64 + 3);
  }
}

auto AsciiContinuousColorResultsPrinter::do_get_extension() -> string {
  return ".txt";
}
auto AsciiContinuousColorResultsPrinter::do_get_format() -> string {
  return "ascii";
}
auto AsciiContinuousColorResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

auto AsciiContinuousColorResultsPrinter::do_with_newline(
  vector<char>::iterator buffer
) -> u64 {
  *buffer = '\n';
  return 1;
}
auto AsciiContinuousColorResultsPrinter::do_with_space(
  vector<char>::iterator buffer
) -> u64 {
  *buffer = ' ';
  return 1;
}
auto AsciiContinuousColorResultsPrinter::do_with_result(
  vector<char>::iterator buffer, u64 result
) -> u64 {
  u64 thread_idx = omp_get_thread_num();
  auto [a, b] = fmt::detail::format_decimal(
    tiny_buffers[thread_idx].data(),
    result,
    static_cast<int>(tiny_buffers[thread_idx].size())
  );
  std::move(a, b, buffer);
  return b - a;
}

}  // namespace sbwt_search
