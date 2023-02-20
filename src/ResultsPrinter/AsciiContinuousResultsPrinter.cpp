#include <algorithm>
#include <bit>
#include <cmath>
#include <ios>
#include <omp.h>

#include <fmt/format.h>
#include <spdlog/fmt/bundled/format.h>

#include "ResultsPrinter/AsciiContinuousResultsPrinter.h"

namespace sbwt_search {

AsciiContinuousResultsPrinter::AsciiContinuousResultsPrinter(
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
  vector<string> &filenames,
  u64 kmer_size,
  u64 threads,
  u64 max_chars_per_batch
):
    Base(
      std::move(results_producer),
      std::move(interval_producer),
      std::move(invalid_chars_producer),
      filenames,
      kmer_size,
      max_chars_in_u64 + 1,
      threads,
      max_chars_per_batch
    ) {
  tiny_buffers.resize(threads);
  for (u64 i = 0; i < threads; ++i) {
    tiny_buffers[i].resize(max_chars_in_u64 + 3);
  }
}

auto AsciiContinuousResultsPrinter::do_with_result(
  vector<char>::iterator buffer, u64 result
) -> u64 {
  u64 thread_idx = omp_get_thread_num();
  auto [a, b] = fmt::detail::format_decimal(
    tiny_buffers[thread_idx].data(),
    result,
    static_cast<int>(tiny_buffers[thread_idx].size())
  );
  std::copy(a, b, buffer);
  return b - a;
}

auto AsciiContinuousResultsPrinter::do_with_not_found(
  vector<char>::iterator buffer
) -> u64 {
  *buffer = '-';
  *(buffer + 1) = '1';
  return 2;
}

auto AsciiContinuousResultsPrinter::do_with_invalid(
  vector<char>::iterator buffer
) -> u64 {
  *buffer = '-';
  *(buffer + 1) = '2';
  return 2;
}

auto AsciiContinuousResultsPrinter::do_with_newline(
  vector<char>::iterator buffer
) -> u64 {
  *buffer = '\n';
  return 1;
}

auto AsciiContinuousResultsPrinter::do_with_space(vector<char>::iterator buffer)
  -> u64 {
  *buffer = ' ';
  return 1;
}

auto AsciiContinuousResultsPrinter::do_get_extension() -> string {
  return ".txt";
}
auto AsciiContinuousResultsPrinter::do_get_format() -> string {
  return "ascii";
}
auto AsciiContinuousResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

}  // namespace sbwt_search