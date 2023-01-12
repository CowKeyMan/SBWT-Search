#include <bit>

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
  ) {}

auto AsciiContinuousResultsPrinter::do_invalid_result() -> void {
  if (!is_at_newline) { write(" "); }
  write("-2");
  is_at_newline = false;
}
auto AsciiContinuousResultsPrinter::do_not_found_result() -> void {
  if (!is_at_newline) { write(" "); }
  write("-1");
  is_at_newline = false;
}
auto AsciiContinuousResultsPrinter::do_result(size_t result) -> void {
  if (!is_at_newline) { write(" "); }
  write(result);
  is_at_newline = false;
}

auto AsciiContinuousResultsPrinter::do_with_newline() -> void {
  write("\n");
  is_at_newline = true;
}

}  // namespace sbwt_search
