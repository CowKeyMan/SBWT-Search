#ifndef BOOL_CONTINUOUS_RESULTS_PRINTER_H
#define BOOL_CONTINUOUS_RESULTS_PRINTER_H

/**
 * @file BoolContinuousResultsPrinter.h
 * @brief Gets results, intervals and list of invalid chars and prints
 * these out to disk based on the given data and filenames.
 */

#include <algorithm>
#include <bit>
#include <memory>

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/InvalidCharsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "BatchObjects/StringSequenceBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using io_utils::ThrowingOfstream;
using std::ios;
using std::shared_ptr;
using std::unique_ptr;

const u8 default_shift_bits = 63;

class BoolContinuousResultsPrinter {
private:
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer;
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer;
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer;
  shared_ptr<IntervalBatch> interval_batch;
  vector<string> filenames;
  u64 chars_index = 0, results_index = 0, line_index = 0;
  u64 invalid_chars_left = 0;
  u64 chars_before_newline_index = 0;
  vector<string>::iterator current_filename;
  shared_ptr<ResultsBatch> results_batch;
  shared_ptr<InvalidCharsBatch> invalid_chars_batch;
  unique_ptr<ThrowingOfstream> out_stream;
  u64 kmer_size;
  vector<char> batch;
  u64 working_bits = 0, working_seq_size = 0;
  u8 shift_bits = default_shift_bits;
  unique_ptr<ThrowingOfstream> seq_size_stream;

public:
  BoolContinuousResultsPrinter(
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
    vector<string> filenames,
    u64 kmer_size,
    u64 max_chars_per_batch
  );

  auto read_and_generate() -> void;

private:
  auto get_batch() -> bool;
  auto process_batch() -> void;
  auto process_file(u64 newlines_before_newfile) -> void;
  auto process_line(u64 chars_before_newline) -> void;
  auto get_invalid_chars_left_first_kmer() -> u64;
  auto process_result(u64 result, bool found, bool valid) -> void;
  auto reset_working_bits() -> void;
  auto shift() -> void;
  auto dump_working_bits() -> void;
  auto get_seq_sizes_extension() -> string;
  auto get_seq_sizes_format() -> string;
  auto get_seq_sizes_version() -> string;
  auto do_invalid_result() -> void;
  auto do_not_found_result() -> void;
  auto do_result(u64 result) -> void;
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
