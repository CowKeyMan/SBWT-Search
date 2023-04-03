#include <string>

#include <fmt/format.h>

#include "ColorResultsPrinter/CsvContinuousColorResultsPrinter.h"
#include "Tools/StdUtils.hpp"

namespace sbwt_search {

using std_utils::copy_advance;

CsvContinuousColorResultsPrinter::CsvContinuousColorResultsPrinter(
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
      2,
      0,
      max_reads_in_buffer,
      write_headers
    ),
    num_colors(num_colors_) {
  row_template.resize(num_colors * 2);
  for (u64 i = 0; i < num_colors; ++i) {
    row_template[i * 2] = '0';
    if (i == num_colors - 1) {
      row_template[i * 2 + 1] = '\n';
    } else {
      row_template[i * 2 + 1] = ',';
    }
  }
}

auto CsvContinuousColorResultsPrinter::do_get_extension() -> string {
  return ".csv";
}

auto CsvContinuousColorResultsPrinter::do_write_file_header(
  ThrowingOfstream &out_stream
) const -> void {
  vector<char> buffer(max_chars_in_u64);
  for (u64 i = 1; i <= num_colors; ++i) {
    auto [a, b] = fmt::detail::format_decimal(
      buffer.data(), i, static_cast<int>(buffer.size())
    );
    out_stream.write(a, b - a);
    if (i == num_colors) {
      out_stream << '\n';
    } else {
      out_stream << ',';
    }
  }
}

auto CsvContinuousColorResultsPrinter::do_print_read(
  vector<u64>::iterator results,
  u64 found_idxs,
  u64 not_found_idxs,
  u64 invalid_idxs,
  vector<char> &buffer,
  u64 &buffer_idx
) -> void {
  std::copy(
    row_template.begin(),
    row_template.end(),
    copy_advance(buffer.begin(), buffer_idx)
  );
  Base::do_print_read(
    results, found_idxs, not_found_idxs, invalid_idxs, buffer, buffer_idx
  );
  buffer_idx += row_template.size();
}

auto CsvContinuousColorResultsPrinter::do_with_newline(
  vector<char>::iterator buffer  // NOLINT (misc-unused-parameters)
) -> u64 {
  return 0;
}
auto CsvContinuousColorResultsPrinter::do_with_result(
  vector<char>::iterator buffer, u64 result
) -> u64 {
  *copy_advance(buffer, result * 2) = '1';
  return 0;
}

}  // namespace sbwt_search
