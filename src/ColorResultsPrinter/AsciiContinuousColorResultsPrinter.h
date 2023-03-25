#ifndef ASCII_CONTINUOUS_COLOR_RESULTS_PRINTER_H
#define ASCII_CONTINUOUS_COLOR_RESULTS_PRINTER_H

/**
 * @file AsciiContinuousColorResultsPrinter.h
 * @brief Outputs ascii results. Color indexes are ordered and space separated,
 * and each read is placed on a new line
 */

#include "ColorResultsPrinter/ContinuousColorResultsPrinter.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class AsciiContinuousColorResultsPrinter:
    public ContinuousColorResultsPrinter<
      AsciiContinuousColorResultsPrinter,
      char> {
  using Base
    = ContinuousColorResultsPrinter<AsciiContinuousColorResultsPrinter, char>;
  friend Base;

public:
  AsciiContinuousColorResultsPrinter(
    u64 stream_id_,
    shared_ptr<SharedBatchesProducer<ColorsIntervalBatch>>
      interval_batch_producer_,
    shared_ptr<SharedBatchesProducer<ReadStatisticsBatch>>
      read_statistics_batch_producer_,
    shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
      results_batch_producer_,
    const vector<string> &filenames_,
    u64 num_colors_,
    double threshold_,
    bool include_not_found_,
    bool include_invalid_
  );

protected:
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

  auto do_with_newline() -> u64;
  auto do_with_space() -> u64;
  auto do_with_result(u64 result) -> u64;
};

}  // namespace sbwt_search

#endif
