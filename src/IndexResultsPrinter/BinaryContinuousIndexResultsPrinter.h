#ifndef BINARY_CONTINUOUS_INDEX_RESULTS_PRINTER_H
#define BINARY_CONTINUOUS_INDEX_RESULTS_PRINTER_H

/**
 * @file BinaryContinuousIndexResultsPrinter.h
 * @brief Inherits ContinuousIndexResultsPrinter and prints out binary values.
 * Not-found characters are reprented with a max_u64, invalids with a max_u64-1
 * and newlines with a max_u64-2.
 */

#include "IndexResultsPrinter/ContinuousIndexResultsPrinter.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class BinaryContinuousIndexResultsPrinter:
    public ContinuousIndexResultsPrinter<
      BinaryContinuousIndexResultsPrinter,
      u64> {
  using Base
    = ContinuousIndexResultsPrinter<BinaryContinuousIndexResultsPrinter, u64>;
  friend Base;

private:
  u64 minus1 = static_cast<u64>(-1), minus2 = static_cast<u64>(-2),
      minus3 = static_cast<u64>(-3);

public:
  BinaryContinuousIndexResultsPrinter(
    u64 stream_id,
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
    vector<string> filenames,
    u64 kmer_size,
    u64 threads,
    u64 max_chars_per_batch,
    u64 max_seqs_per_batch,
    bool write_headers
  );

  auto static get_bits_per_element() -> u64;
  auto static get_bits_per_seq() -> u64;

protected:
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

  [[nodiscard]] auto
  do_with_result(vector<u64>::iterator buffer, u64 result) const -> u64;
  [[nodiscard]] auto do_with_not_found(vector<u64>::iterator buffer) const
    -> u64;
  [[nodiscard]] auto do_with_invalid(vector<u64>::iterator buffer) const -> u64;
  [[nodiscard]] auto do_with_newline(vector<u64>::iterator buffer) const -> u64;
};

}  // namespace sbwt_search

#endif
