#include <string>

#include "ColorResultsPrinter/AsciiContinuousColorResultsPrinter.h"
#include "itoa/jeaiii_to_text.h"

namespace sbwt_search {

AsciiContinuousColorResultsPrinter::AsciiContinuousColorResultsPrinter(
  u64 stream_id_,
  shared_ptr<SharedBatchesProducer<SeqStatisticsBatch>>
    seq_statistics_batch_producer_,
  shared_ptr<SharedBatchesProducer<ColorsBatch>> colors_batch_producer_,
  const vector<string> &filenames_,
  u64 num_colors_,
  double threshold_,
  bool include_not_found_,
  bool include_invalid_,
  u64 threads,
  u64 max_reads_per_batch,
  bool write_headers
):
    Base(
      stream_id_,
      std::move(seq_statistics_batch_producer_),
      std::move(colors_batch_producer_),
      filenames_,
      num_colors_,
      threshold_,
      include_not_found_,
      include_invalid_,
      threads,
      get_bits_per_read(num_colors_) / bits_in_byte,
      max_reads_per_batch,
      write_headers
    ) {}

auto AsciiContinuousColorResultsPrinter::get_bits_per_read(u64 num_colors)
  -> u64 {
  u64 num_colors_copy = num_colors;
  const u64 bits_required_per_whitespace = bits_in_byte;
  u64 bits_required_per_character = 0;
  const u64 base_10 = 10;
  while (num_colors_copy > 0) {
    num_colors_copy /= base_10;
    ++bits_required_per_character;
  }
  bits_required_per_character *= bits_in_byte;
  return bits_required_per_character * num_colors
    + bits_required_per_whitespace;
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
  auto buffer_end = jeaiii::to_text_from_integer(buffer.base(), result);
  return buffer_end - buffer.base();
}

}  // namespace sbwt_search
