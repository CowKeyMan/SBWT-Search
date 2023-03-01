#ifndef CONTINUOUS_COLOR_SEARCH_RESULTS_PRINTER_H
#define CONTINUOUS_COLOR_SEARCH_RESULTS_PRINTER_H

/**
 * @file ContinuousColorSearchResultsPrinter.h
 * @brief
 */

#include <ios>
#include <limits>
#include <memory>

#include "BatchObjects/ColorSearchResultsBatch.h"
#include "BatchObjects/ColorsIntervalBatch.h"
#include "BatchObjects/ReadStatisticsBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "fmt/core.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using fmt::format;
using io_utils::ThrowingOfstream;
using log_utils::Logger;
using std::ios;
using std::numeric_limits;
using std::shared_ptr;
using std::unique_ptr;

class ContinuousColorSearchResultsPrinter {
private:
  /* auto impl() -> TImplementation & { */
  /*   return static_cast<TImplementation &>(*this); */
  /* } */
  auto impl() -> ContinuousColorSearchResultsPrinter & { return *this; }
  shared_ptr<SharedBatchesProducer<ColorsIntervalBatch>>
    interval_batch_producer;
  shared_ptr<SharedBatchesProducer<ReadStatisticsBatch>>
    read_statistics_batch_producer;
  shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
    results_batch_producer;
  shared_ptr<ColorsIntervalBatch> interval_batch;
  shared_ptr<ReadStatisticsBatch> read_statistics_batch;
  shared_ptr<ColorSearchResultsBatch> results_batch;
  vector<string> filenames;
  vector<string>::iterator current_filename;
  unique_ptr<ThrowingOfstream> out_stream;
  u64 num_colors;
  double threshold;
  vector<u64> previous_last_results;
  u64 previous_last_found_idx = 0;
  u64 previous_last_not_found_idx = 0;
  u64 previous_last_invalid_idx = 0;

public:
  ContinuousColorSearchResultsPrinter(
    shared_ptr<SharedBatchesProducer<ColorsIntervalBatch>>
      interval_batch_producer_,
    shared_ptr<SharedBatchesProducer<ReadStatisticsBatch>>
      read_statistics_batch_producer_,
    shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
      results_batch_producer_,
    const vector<string> &filenames_,
    u64 num_colors_,
    double threshold_
  ):
      interval_batch_producer(std::move(interval_batch_producer_)),
      read_statistics_batch_producer(std::move(read_statistics_batch_producer_)
      ),
      results_batch_producer(std::move(results_batch_producer_)),
      filenames(filenames_),
      num_colors(num_colors_),
      threshold(threshold_),
      previous_last_results(num_colors_, 0){};

  auto read_and_generate() -> void {
    current_filename = filenames.begin();
    if (current_filename == filenames.end()) { return; }
    impl().do_start_next_file();
    for (u64 batch_id = 0; get_batch(); ++batch_id) {
      Logger::log_timed_event(
        "ColorSearchResultsPrinter",
        Logger::EVENT_STATE::START,
        format("batch {}", batch_id)
      );
      process_batch();
      Logger::log_timed_event(
        "ColorSearchResultsPrinter",
        Logger::EVENT_STATE::STOP,
        format("batch {}", batch_id)
      );
    }
    impl().do_at_file_end();
  }

private:
  auto get_batch() -> bool {
    return (static_cast<u64>(*interval_batch_producer >> interval_batch)
            & static_cast<u64>(
              *read_statistics_batch_producer >> read_statistics_batch
            )
            & static_cast<u64>(*results_batch_producer >> results_batch))
      > 0;
  }

protected:
  auto do_get_extension() -> string { return ".out"; }  // TODO: change
  auto do_get_format() -> string { return "format"; }
  auto do_get_version() -> string { return "version"; }

  auto do_start_next_file() -> void {
    if (current_filename != filenames.begin()) { impl().do_at_file_end(); }
    impl().do_open_next_file(*current_filename);
    impl().do_write_file_header();
    current_filename = next(current_filename);
  }
  auto do_at_file_end() -> void {}
  auto do_open_next_file(const string &filename) -> void {
    out_stream = make_unique<ThrowingOfstream>(
      filename + impl().do_get_extension(), ios::binary | ios::out
    );
  }
  auto do_write_file_header() -> void {
    out_stream->write_string_with_size(impl().do_get_format());
    out_stream->write_string_with_size(impl().do_get_version());
  }
  auto process_batch() -> void {
    auto &results = *results_batch->results;
    auto &found_idxs = read_statistics_batch->found_idxs;
    auto &not_found_idxs = read_statistics_batch->not_found_idxs;
    auto &invalid_idxs = read_statistics_batch->invalid_idxs;
    auto &wbnrs = *interval_batch->warps_before_new_read;
    auto &rbnfs = interval_batch->reads_before_newfile;
    if (rbnfs[0] == 0) {
      impl().do_print_read(
        0, found_idxs[0] + not_found_idxs[0] + invalid_idxs[0]
      );
    } else {
      // Fill in from previous batch (read is continued)
      found_idxs[0] += previous_last_found_idx;
      not_found_idxs[0] += previous_last_not_found_idx;
      invalid_idxs[0] += previous_last_invalid_idx;
#pragma omp simd
      for (int i = 0; i < num_colors; ++i) {
        results[i] += previous_last_results[i];
      }
    }
    u64 rbnf_idx = 0;
    for (u64 wbnr_idx = 0; wbnr_idx < wbnrs.size(); ++wbnr_idx) {
      const auto &read_idx = wbnr_idx;
      while (rbnfs[rbnf_idx] == read_idx) {
        impl().do_start_next_file();
        ++rbnf_idx;
      }
      if (wbnrs[wbnr_idx + 1] == numeric_limits<u64>::max()) {
        previous_last_found_idx = found_idxs[read_idx];
        previous_last_not_found_idx = not_found_idxs[read_idx];
        previous_last_invalid_idx = invalid_idxs[read_idx];
#pragma omp simd
        for (int i = 0; i < num_colors; ++i) {
          previous_last_results[i] = results[wbnrs[wbnr_idx] * num_colors + i];
        }
        break;
      }
      impl().do_print_read(
        wbnrs[wbnr_idx] * num_colors,
        found_idxs[read_idx] + not_found_idxs[read_idx] + invalid_idxs[read_idx]
      );
    }
  }
  auto do_print_read(u64 results_idx, u64 read_size) -> void {
    const auto &results = *results_batch->results;
    u64 minimum_found
      = static_cast<u64>(static_cast<double>(read_size) * threshold);
    for (u64 res_idx = results_idx; res_idx < results_idx + num_colors;
         ++res_idx) {
      if (results[res_idx] > minimum_found) {
        *out_stream << results[res_idx];
        if (res_idx + 1 < results_idx + num_colors) { *out_stream << " "; }
      }
    }
    *out_stream << "\n";
  }
};

}  // namespace sbwt_search

#endif
