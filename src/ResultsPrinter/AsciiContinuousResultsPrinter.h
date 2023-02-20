#ifndef ASCII_CONTINUOUS_RESULTS_PRINTER_H
#define ASCII_CONTINUOUS_RESULTS_PRINTER_H

/**
 * @file AsciiContinuousResultsPrinter.h
 * @brief Inherits ContinuousResultsPrinter and prints out ascii values
 */

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"

namespace sbwt_search {

const u64 max_chars_in_u64 = 20;

class AsciiContinuousResultsPrinter:
    public ContinuousResultsPrinter<AsciiContinuousResultsPrinter, char> {
  using Base = ContinuousResultsPrinter<AsciiContinuousResultsPrinter, char>;
  friend Base;

private:
  vector<vector<char>> tiny_buffers;

public:
  AsciiContinuousResultsPrinter(
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
    vector<string> &filenames,
    u64 kmer_size,
    u64 threads,
    u64 max_chars_per_batch_
  );

protected:
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

  auto do_with_result(vector<char>::iterator buffer, u64 result) -> u64;
  auto do_with_not_found(vector<char>::iterator buffer) -> u64;
  auto do_with_invalid(vector<char>::iterator buffer) -> u64;
  auto do_with_newline(vector<char>::iterator buffer) -> u64;
  auto do_with_space(vector<char>::iterator buffer) -> u64;
};

}  // namespace sbwt_search

#endif