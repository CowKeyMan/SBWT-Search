#ifndef ASCII_CONTINUOUS_RESULTS_PRINTER_H
#define ASCII_CONTINUOUS_RESULTS_PRINTER_H

/**
 * @file AsciiContinuousResultsPrinter.h
 * @brief Inherits ContinuousResultsPrinter and prints out ascii values
 */

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"

namespace sbwt_search {

class AsciiContinuousResultsPrinter:
  public ContinuousResultsPrinter<AsciiContinuousResultsPrinter> {
  using Base = ContinuousResultsPrinter<AsciiContinuousResultsPrinter>;
  friend Base;

private:
  bool is_at_newline = true;

public:
  AsciiContinuousResultsPrinter(
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
    vector<string> &filenames,
    uint kmer_size
  );

protected:
  auto do_invalid_result() -> void;
  auto do_not_found_result() -> void;
  auto do_result(size_t result) -> void;
  auto do_with_newline() -> void;
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;
};

}  // namespace sbwt_search

#endif
