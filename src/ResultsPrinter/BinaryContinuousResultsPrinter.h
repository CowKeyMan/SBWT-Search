#ifndef BINARY_CONTINUOUS_RESULTS_PRINTER_H
#define BINARY_CONTINUOUS_RESULTS_PRINTER_H

/**
 * @file BinaryContinuousResultsPrinter.h
 * @brief Inherits ContinuousResultsPrinter and prints out binary values
 */

#include "ResultsPrinter/ContinuousResultsPrinter.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class BinaryContinuousResultsPrinter:
  public ContinuousResultsPrinter<BinaryContinuousResultsPrinter> {
  using Base = ContinuousResultsPrinter<BinaryContinuousResultsPrinter>;
  friend Base;

private:
  u64 minus1 = static_cast<u64>(-1), minus2 = static_cast<u64>(-2),
      minus3 = static_cast<u64>(-3);

public:
  BinaryContinuousResultsPrinter(
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
