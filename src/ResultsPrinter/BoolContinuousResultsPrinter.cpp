#include <algorithm>
#include <bit>
#include <memory>

#include "ResultsPrinter/BoolContinuousResultsPrinter.h"
#include "Tools/IOUtils.h"
#include "Tools/MathUtils.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

using math_utils::round_up;
using std::bit_cast;
using std::ios;

auto BoolContinuousResultsPrinter::do_get_extension() -> string {
  return ".bool";
}
auto BoolContinuousResultsPrinter::get_seq_sizes_extension() -> string {
  return ".seqsizes";
}
auto BoolContinuousResultsPrinter::do_get_format() -> string { return "bool"; }
auto BoolContinuousResultsPrinter::do_get_version() -> string { return "v1.0"; }
auto BoolContinuousResultsPrinter::get_seq_sizes_format() -> string {
  return "seq_sizes";
}
auto BoolContinuousResultsPrinter::get_seq_sizes_version() -> string {
  return "v1.0";
}

BoolContinuousResultsPrinter::BoolContinuousResultsPrinter(
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
  vector<string> &filenames,
  uint kmer_size,
  u64 max_chars_per_batch
):
    Base(
      std::move(results_producer),
      std::move(interval_producer),
      std::move(invalid_chars_producer),
      filenames,
      kmer_size
    ),
    batch(round_up<u64>(max_chars_per_batch, sizeof(u64)) / sizeof(u64)) {
  reset_working_bits();
}

auto BoolContinuousResultsPrinter::do_invalid_result() -> void { shift(); }

auto BoolContinuousResultsPrinter::do_not_found_result() -> void { shift(); }

// NOLINTNEXTLINE(misc-unused-parameters)
auto BoolContinuousResultsPrinter::do_result(size_t result) -> void {
  working_bits |= (1ULL << shift_bits);
  shift();
}

auto BoolContinuousResultsPrinter::do_with_newline() -> void {
  seq_size_stream->write(bit_cast<char *>(&working_seq_size), sizeof(u64));
  working_seq_size = 0;
}

auto BoolContinuousResultsPrinter::do_at_file_end() -> void {
  dump_working_bits();
  reset_working_bits();
}

auto BoolContinuousResultsPrinter::do_open_next_file() -> void {
  ContinuousResultsPrinter::do_open_next_file();
  seq_size_stream = make_unique<ThrowingOfstream>(
    get_current_filename() + get_seq_sizes_extension(), ios::binary | ios::out
  );
}

auto BoolContinuousResultsPrinter::do_start_next_file() -> void {
  Base::do_start_next_file();
}

auto BoolContinuousResultsPrinter::reset_working_bits() -> void {
  working_bits = 0;
  shift_bits = default_shift_bits;
}

auto BoolContinuousResultsPrinter::shift() -> void {
  ++working_seq_size;
  if (shift_bits == 0) {
    dump_working_bits();
    reset_working_bits();
    return;
  }
  --shift_bits;
}

auto BoolContinuousResultsPrinter::dump_working_bits() -> void {
  get_ostream().write(bit_cast<char *>(&working_bits), sizeof(u64));
}

auto BoolContinuousResultsPrinter::do_write_file_header() -> void {
  Base::do_write_file_header();
  get_ostream().write_string_with_size(get_seq_sizes_format());
  get_ostream().write_string_with_size(get_seq_sizes_version());
}

}  // namespace sbwt_search
