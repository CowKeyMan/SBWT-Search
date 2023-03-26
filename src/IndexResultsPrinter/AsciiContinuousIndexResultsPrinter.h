#ifndef ASCII_CONTINUOUS_INDEX_RESULTS_PRINTER_H
#define ASCII_CONTINUOUS_INDEX_RESULTS_PRINTER_H

/**
 * @file AsciiContinuousIndexResultsPrinter.h
 * @brief Inherits ContinuousIndexResultsPrinter and prints out ascii values
 */

#include "IndexResultsPrinter/ContinuousIndexResultsPrinter.hpp"

namespace sbwt_search {

class AsciiContinuousIndexResultsPrinter:
    public ContinuousIndexResultsPrinter<
      AsciiContinuousIndexResultsPrinter,
      char> {
  using Base
    = ContinuousIndexResultsPrinter<AsciiContinuousIndexResultsPrinter, char>;
  friend Base;

private:
  vector<vector<char>> tiny_buffers;

public:
  AsciiContinuousIndexResultsPrinter(
    u64 stream_id,
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer,
    vector<string> filenames_,
    u64 kmer_size,
    u64 threads,
    u64 max_chars_per_batch,
    u64 max_reads_per_batch
  );

protected:
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

  [[nodiscard]] auto do_with_result(vector<char>::iterator buffer, u64 result)
    -> u64;
  [[nodiscard]] auto do_with_not_found(vector<char>::iterator buffer) const
    -> u64;
  [[nodiscard]] auto do_with_invalid(vector<char>::iterator buffer) const
    -> u64;
  [[nodiscard]] auto do_with_space(vector<char>::iterator buffer) const -> u64;
  [[nodiscard]] auto do_with_newline(vector<char>::iterator buffer) const
    -> u64;
};

}  // namespace sbwt_search

#endif
