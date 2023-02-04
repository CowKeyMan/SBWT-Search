#include <bit>

#include <fmt/format.h>

#include "ResultsPrinter/AsciiContinuousResultsPrinter.h"

namespace sbwt_search {

auto AsciiContinuousResultsPrinter::do_get_extension() -> string {
  return ".txt";
}
auto AsciiContinuousResultsPrinter::do_get_format() -> string {
  return "ascii";
}
auto AsciiContinuousResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

AsciiContinuousResultsPrinter::AsciiContinuousResultsPrinter(
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
  vector<string> &filenames,
  uint kmer_size
):
    Base(
      std::move(results_producer),
      std::move(interval_producer),
      std::move(invalid_chars_producer),
      filenames,
      kmer_size
    ),
    buffer(max_characters_in_u64, '\0') {}

auto AsciiContinuousResultsPrinter::do_invalid_result() -> void {
  if (!is_at_newline) { get_ostream() << " "; }
  get_ostream() << "-2";
  is_at_newline = false;
}
auto AsciiContinuousResultsPrinter::do_not_found_result() -> void {
  if (!is_at_newline) { get_ostream() << " "; }
  get_ostream() << "-1";
  is_at_newline = false;
}

auto fast_int_to_string(int64_t x, char *buffer) -> void {
  int64_t i = 0;
  // Write the digits in reverse order (reversed back at the end)
  if (x == 0) {
    buffer[0] = '0';
    i = 1;
  } else {
    while (x > 0) {
      buffer[i++] = '0' + (x % 10);
      x /= 10;
    }
  }
  std::reverse(buffer, buffer + i);
  buffer[i] = '\0';
}

auto AsciiContinuousResultsPrinter::do_result(size_t result) -> void {
  if (!is_at_newline) { get_ostream() << " "; }
  get_ostream() << fmt::format_int(result).c_str();
  /* get_ostream().write(result); */
  /* fast_int_to_string(result, buffer); */
  /* get_ostream().write(buffer); */
  is_at_newline = false;
}

auto AsciiContinuousResultsPrinter::do_with_newline() -> void {
  get_ostream() << "\n";
  is_at_newline = true;
}

}  // namespace sbwt_search
