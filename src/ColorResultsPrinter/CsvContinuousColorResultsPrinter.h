#ifndef CSV_CONTINUOUS_COLOR_RESULTS_PRINTER_H
#define CSV_CONTINUOUS_COLOR_RESULTS_PRINTER_H

/**
 * @file CsvContinuousColorResultsPrinter.h
 * @brief Outputs csv results. Color indexes are ordered and space separated,
 * and each seq is placed on a new line
 */

#include "ColorResultsPrinter/ContinuousColorResultsPrinter.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class CsvContinuousColorResultsPrinter:
    public ContinuousColorResultsPrinter<
      CsvContinuousColorResultsPrinter,
      char> {
  using Base
    = ContinuousColorResultsPrinter<CsvContinuousColorResultsPrinter, char>;
  friend Base;

  u64 num_colors;
  vector<char> row_template;

public:
  CsvContinuousColorResultsPrinter(
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

  auto do_print_seq(
    vector<u64>::iterator results,
    u64 found_idxs,
    u64 not_found_idxs,
    u64 invalid_idxs,
    vector<char> &buffer,
    u64 &buffer_idx
  ) -> void;

  auto do_write_file_header(ThrowingOfstream &out_stream) const -> void;
  auto do_with_newline(vector<char>::iterator buffer) -> u64;
  auto do_with_result(vector<char>::iterator buffer, u64 result) -> u64;
};

}  // namespace sbwt_search

#endif
