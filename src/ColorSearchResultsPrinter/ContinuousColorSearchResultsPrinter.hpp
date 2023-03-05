#ifndef CONTINUOUS_COLOR_SEARCH_RESULTS_PRINTER_HPP
#define CONTINUOUS_COLOR_SEARCH_RESULTS_PRINTER_HPP

/**
 * @file ContinuousColorSearchResultsPrinter.hpp
 * @brief Prints out the color results
 */

#include <cmath>
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
  u64 previous_last_found_idx = numeric_limits<u64>::max();
  u64 previous_last_not_found_idxs = numeric_limits<u64>::max();
  u64 previous_last_invalid_idxs = numeric_limits<u64>::max();
  bool printed_last_read = false;
  u64 include_not_found;
  u64 include_invalid;

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
    double threshold_,
    bool include_not_found_,
    bool include_invalid_
  ):
      interval_batch_producer(std::move(interval_batch_producer_)),
      read_statistics_batch_producer(std::move(read_statistics_batch_producer_)
      ),
      results_batch_producer(std::move(results_batch_producer_)),
      filenames(filenames_),
      num_colors(num_colors_),
      threshold(threshold_),
      previous_last_results(num_colors_, 0),
      include_not_found(static_cast<u64>(include_not_found_)),
      include_invalid(static_cast<u64>(include_invalid_)){};

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
      printed_last_read = false;
      process_batch();
      Logger::log_timed_event(
        "ColorSearchResultsPrinter",
        Logger::EVENT_STATE::STOP,
        format("batch {}", batch_id)
      );
    }
    if (!printed_last_read) {
      impl().do_print_read(
        previous_last_results.begin(),
        previous_last_found_idx,
        previous_last_not_found_idxs,
        previous_last_invalid_idxs
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
  auto do_get_extension() -> string { return ".txt"; }  // TODO: change
  auto do_get_format() -> string { return "ascii"; }
  auto do_get_version() -> string { return "v1.0"; }

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
        previous_last_results.begin(),
        previous_last_found_idx,
        previous_last_not_found_idxs,
        previous_last_invalid_idxs
      );
      printed_last_read = true;
    } else {
      // Fill in from previous batch (read is continued)
      found_idxs[0] += previous_last_found_idx;
      not_found_idxs[0] += previous_last_not_found_idxs;
      invalid_idxs[0] += previous_last_invalid_idxs;
#pragma omp simd
      for (int i = 0; i < num_colors; ++i) {
        results[i] += previous_last_results[i];
      }
    }
    u64 rbnf_idx = 0;
    u64 wbnr = 0;
    for (u64 wbnr_idx = 0; wbnr_idx < wbnrs.size(); ++wbnr_idx) {
      const auto &read_idx = wbnr_idx;
      while (rbnfs[rbnf_idx] == read_idx) {
        impl().do_start_next_file();
        ++rbnf_idx;
      }
      if (wbnrs[wbnr_idx] == numeric_limits<u64>::max()) {
        previous_last_found_idx = found_idxs[read_idx];
        previous_last_not_found_idxs = not_found_idxs[read_idx];
        previous_last_invalid_idxs = invalid_idxs[read_idx];
#pragma omp simd
        for (int i = 0; i < num_colors; ++i) {
          previous_last_results[i] = results[wbnr * num_colors + i];
        }
        break;
      }
      impl().do_print_read(
        results.begin() + wbnr * num_colors,
        found_idxs[read_idx],
        not_found_idxs[read_idx],
        invalid_idxs[read_idx]
      );
      wbnr = wbnrs[wbnr_idx];
    }
  }
  auto do_print_read(
    vector<u64>::iterator results,
    u64 found_idxs,
    u64 not_found_idxs,
    u64 invalid_idxs
  ) -> void {
    u64 read_size = found_idxs + include_not_found * not_found_idxs
      + include_invalid * invalid_idxs;
    u64 minimum_found
      = static_cast<u64>(std::ceil(static_cast<double>(read_size) * threshold));
    for (u64 color_idx = 0; color_idx < num_colors; ++color_idx, ++results) {
      if (*results >= minimum_found && minimum_found > 0) {
        *out_stream << color_idx;
        if (color_idx + 1 < num_colors) { *out_stream << " "; }
      }
    }
    *out_stream << "\n";
  }
};

}  // namespace sbwt_search

#endif
