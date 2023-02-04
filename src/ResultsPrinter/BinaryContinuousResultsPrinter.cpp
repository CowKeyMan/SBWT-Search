#include <bit>

#include "ResultsPrinter/BinaryContinuousResultsPrinter.h"

namespace sbwt_search {

using std::bit_cast;

auto BinaryContinuousResultsPrinter::do_get_extension() -> string {
  return ".bin";
}
auto BinaryContinuousResultsPrinter::do_get_format() -> string {
  return "binary";
}
auto BinaryContinuousResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

BinaryContinuousResultsPrinter::BinaryContinuousResultsPrinter(
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

auto BinaryContinuousResultsPrinter::do_invalid_result() -> void {
  get_ostream().write(bit_cast<char *>(&minus2), sizeof(u64));
}
auto BinaryContinuousResultsPrinter::do_not_found_result() -> void {
  get_ostream().write(bit_cast<char *>(&minus1), sizeof(u64));
}
auto BinaryContinuousResultsPrinter::do_result(size_t result) -> void {
  get_ostream().write(bit_cast<char *>(&result), sizeof(u64));
}
auto BinaryContinuousResultsPrinter::do_with_newline() -> void {
  get_ostream().write(bit_cast<char *>(&minus3), sizeof(u64));
}

}  // namespace sbwt_search
