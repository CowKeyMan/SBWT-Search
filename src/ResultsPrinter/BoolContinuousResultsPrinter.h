#ifndef BOOL_CONTINUOUS_RESULTS_PRINTER_H
#define BOOL_CONTINUOUS_RESULTS_PRINTER_H

/**
 * @file BoolContinuousResultsPrinter.h
 * @brief Inherits ContinuousResultsPrinter and prints out boolean bitvectors
 */

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"

namespace sbwt_search {

using std::unique_ptr;

const u8 default_shift_bits = 63;

class BoolContinuousResultsPrinter:
    public ContinuousResultsPrinter<BoolContinuousResultsPrinter> {
  using Base = ContinuousResultsPrinter<BoolContinuousResultsPrinter>;
  friend Base;

private:
  vector<char> batch;
  u64 working_bits = 0, working_seq_size = 0;
  u8 shift_bits = default_shift_bits;
  unique_ptr<ThrowingOfstream> seq_size_stream;
  auto get_seq_sizes_extension() -> string;
  auto get_seq_sizes_format() -> string;
  auto get_seq_sizes_version() -> string;

public:
  BoolContinuousResultsPrinter(
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
    vector<string> &filenames,
    uint kmer_size,
    u64 max_chars_per_batch
  );

private:
  auto reset_working_bits() -> void;
  auto shift() -> void;
  auto dump_working_bits() -> void;

protected:
  auto do_invalid_result() -> void;
  auto do_not_found_result() -> void;
  auto do_result(size_t result) -> void;
  auto do_with_newline() -> void;
  auto do_at_file_end() -> void;
  auto do_open_next_file() -> void;
  auto do_start_next_file() -> void;
  auto do_write_file_header() -> void;
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;
};

}  // namespace sbwt_search

#endif
