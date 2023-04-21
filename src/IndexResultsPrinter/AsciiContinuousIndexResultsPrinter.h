#ifndef ASCII_CONTINUOUS_INDEX_RESULTS_PRINTER_H
#define ASCII_CONTINUOUS_INDEX_RESULTS_PRINTER_H

/**
 * @file AsciiContinuousIndexResultsPrinter.h
 * @brief Inherits ContinuousIndexResultsPrinter and prints out ascii values.
 * Indexes are printed as their space separated ASCII values, with a newline
 * character separating the reads. Not-found values are printed as '-1' and
 * invalid values are printed as '-2'. When calculating memory reservations for
 * this class, we use the max_index to see how many characters we really need
 * per index, rather than the maximum needed for the maximum u64. This saves us
 * a lot of space.
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
    u64 max_reads_per_batch,
    bool write_headers,
    u64 max_index
  );

  static auto get_bits_per_element(u64 max_index) -> u64;

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
