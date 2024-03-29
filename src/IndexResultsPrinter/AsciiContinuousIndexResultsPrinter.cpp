#include <fmt/format.h>

#include "IndexResultsPrinter/AsciiContinuousIndexResultsPrinter.h"
#include "Tools/TypeDefinitions.h"
#include "itoa/jeaiii_to_text.h"

namespace sbwt_search {

AsciiContinuousIndexResultsPrinter::AsciiContinuousIndexResultsPrinter(
  u64 stream_id,
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
  vector<string> filenames,
  u64 kmer_size,
  u64 threads,
  u64 max_chars_per_batch,
  u64 max_seqs_per_batch,
  bool write_headers,
  u64 max_index
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
      max_seqs_per_batch,
      get_bits_per_element(max_index) / bits_in_byte,
      0,
      write_headers
    ) {}

auto AsciiContinuousIndexResultsPrinter::get_bits_per_element(u64 max_index)
  -> u64 {
  const u64 bits_required_per_whitespace = bits_in_byte;
  u64 bits_required_per_character = 0;
  const u64 base_10 = 10;
  while (max_index > 0) {
    max_index /= base_10;
    ++bits_required_per_character;
  }
  bits_required_per_character *= bits_in_byte;
  return bits_required_per_character + bits_required_per_whitespace;
}

auto AsciiContinuousIndexResultsPrinter::do_with_result(
  vector<char>::iterator buffer, u64 result
) -> u64 {
  auto buffer_end = jeaiii::to_text_from_integer(buffer.base(), result);
  return buffer_end - buffer.base();
}

auto AsciiContinuousIndexResultsPrinter::do_with_not_found(
  vector<char>::iterator buffer
) const -> u64 {
  *buffer = '-';
  *(buffer + 1) = '1';
  return 2;
}

auto AsciiContinuousIndexResultsPrinter::do_with_invalid(
  vector<char>::iterator buffer
) const -> u64 {
  *buffer = '-';
  *(buffer + 1) = '2';
  return 2;
}

auto AsciiContinuousIndexResultsPrinter::do_with_space(
  vector<char>::iterator buffer
) const -> u64 {
  *buffer = ' ';
  return 1;
}

auto AsciiContinuousIndexResultsPrinter::do_with_newline(
  vector<char>::iterator buffer
) const -> u64 {
  *buffer = '\n';
  return 1;
}

auto AsciiContinuousIndexResultsPrinter::do_get_extension() -> string {
  return ".txt";
}
auto AsciiContinuousIndexResultsPrinter::do_get_format() -> string {
  return "ascii";
}
auto AsciiContinuousIndexResultsPrinter::do_get_version() -> string {
  return "v1.0";
}

}  // namespace sbwt_search
