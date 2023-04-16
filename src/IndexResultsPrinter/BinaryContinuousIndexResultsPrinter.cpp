#include <bit>

#include "IndexResultsPrinter/BinaryContinuousIndexResultsPrinter.h"

namespace sbwt_search {

BinaryContinuousIndexResultsPrinter::BinaryContinuousIndexResultsPrinter(
  u64 stream_id,
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
  vector<string> filenames,
  u64 kmer_size,
  u64 threads,
  u64 max_chars_per_batch,
  u64 max_reads_per_batch,
  bool write_headers
):
    Base(
      stream_id,
      std::move(results_producer),
      std::move(interval_producer),
      std::move(invalid_chars_producer),
      std::move(filenames),
      kmer_size,
      threads,
      max_chars_per_batch,
      max_reads_per_batch,
      1,
      1,
      write_headers
    ) {}

auto BinaryContinuousIndexResultsPrinter::get_bits_per_element() -> u64 {
  return u64_bits;
}

auto BinaryContinuousIndexResultsPrinter::get_bits_per_read() -> u64 {
  return u64_bits;
}

auto BinaryContinuousIndexResultsPrinter::do_get_extension() -> string {
  return ".bin";
}
auto BinaryContinuousIndexResultsPrinter::do_get_format() -> string {
  return "binary";
}
auto BinaryContinuousIndexResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

auto BinaryContinuousIndexResultsPrinter::do_with_result(
  vector<u64>::iterator buffer, u64 result
) const -> u64 {
  *buffer = result;
  return 1;
}

auto BinaryContinuousIndexResultsPrinter::do_with_not_found(
  vector<u64>::iterator buffer
) const -> u64 {
  *buffer = minus1;
  return 1;
}

auto BinaryContinuousIndexResultsPrinter::do_with_invalid(
  vector<u64>::iterator buffer
) const -> u64 {
  *buffer = minus2;
  return 1;
}

auto BinaryContinuousIndexResultsPrinter::do_with_newline(
  vector<u64>::iterator buffer
) const -> u64 {
  *buffer = minus3;
  return 1;
}

}  // namespace sbwt_search
