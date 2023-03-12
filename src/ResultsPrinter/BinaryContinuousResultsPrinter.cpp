#include <bit>

#include "ResultsPrinter/BinaryContinuousResultsPrinter.h"

namespace sbwt_search {

BinaryContinuousResultsPrinter::BinaryContinuousResultsPrinter(
  u64 stream_id,
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
  vector<string> filenames,
  u64 kmer_size,
  u64 threads,
  u64 max_chars_per_batch
):
    Base(
      stream_id,
      std::move(results_producer),
      std::move(interval_producer),
      std::move(invalid_chars_producer),
      std::move(filenames),
      kmer_size,
      1,
      threads,
      max_chars_per_batch
    ) {}

auto BinaryContinuousResultsPrinter::do_get_extension() -> string {
  return ".bin";
}
auto BinaryContinuousResultsPrinter::do_get_format() -> string {
  return "binary";
}
auto BinaryContinuousResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

auto BinaryContinuousResultsPrinter::do_with_result(
  vector<u64>::iterator buffer, u64 result
) -> u64 {
  *buffer = result;
  return 1;
}

auto BinaryContinuousResultsPrinter::do_with_not_found(
  vector<u64>::iterator buffer
) -> u64 {
  *buffer = minus1;
  return 1;
}

auto BinaryContinuousResultsPrinter::do_with_invalid(
  vector<u64>::iterator buffer
) -> u64 {
  *buffer = minus2;
  return 1;
}

auto BinaryContinuousResultsPrinter::do_with_newline(
  vector<u64>::iterator buffer
) -> u64 {
  *buffer = minus3;
  return 1;
}

}  // namespace sbwt_search
