#ifndef BINARY_CONTINUOUS_COLOR_RESULTS_PRINTER_H
#define BINARY_CONTINUOUS_COLOR_RESULTS_PRINTER_H

/**
 * @file BinaryContinuousColorResultsPrinter.h
 * @brief Outputs binary results. Color indexes are ordered and space separated,
 * and each seq is placed on a new line
 */

#include "ColorResultsPrinter/ContinuousColorResultsPrinter.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class BinaryContinuousColorResultsPrinter:
    public ContinuousColorResultsPrinter<
      BinaryContinuousColorResultsPrinter,
      u64> {
  using Base
    = ContinuousColorResultsPrinter<BinaryContinuousColorResultsPrinter, u64>;
  friend Base;

public:
  BinaryContinuousColorResultsPrinter(
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
    u64 max_seqs_per_batch,
    bool write_headers
  );

  auto static get_bits_per_seq(u64 num_colors) -> u64;

protected:
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

  auto do_with_newline(vector<u64>::iterator buffer) -> u64;
  auto do_with_result(vector<u64>::iterator buffer, u64 result) -> u64;
};

}  // namespace sbwt_search

#endif
