#ifndef CONTINUOUS_COLOR_RESULTS_PRINTER_HPP
#define CONTINUOUS_COLOR_RESULTS_PRINTER_HPP

/**
 * @file ContinuousColorResultsPrinter.hpp
 * @brief Prints out the color results
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>

#include "BatchObjects/ColorSearchResultsBatch.h"
#include "BatchObjects/ColorsIntervalBatch.h"
#include "BatchObjects/ReadStatisticsBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/OmpLock.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/StdUtils.hpp"
#include "fmt/core.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using fmt::format;
using io_utils::ThrowingOfstream;
using log_utils::Logger;
using std::bit_cast;
using std::ios;
using std::numeric_limits;
using std::shared_ptr;
using std::unique_ptr;
using std_utils::copy_advance;
using threading_utils::OmpLock;

template <class TImplementation, class Buffer_t>
class ContinuousColorResultsPrinter {
private:
  auto impl() -> TImplementation & {
    return static_cast<TImplementation &>(*this);
  }
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
  u64 num_colors;
  double threshold;
  vector<u64> previous_last_results;
  u64 previous_last_found_idx = numeric_limits<u64>::max();
  u64 previous_last_not_found_idxs = numeric_limits<u64>::max();
  u64 previous_last_invalid_idxs = numeric_limits<u64>::max();
  bool printed_last_read = false;
  u64 include_not_found;
  u64 include_invalid;
  vector<vector<Buffer_t>> buffers;
  u64 threads;
  u64 stream_id;
  vector<OmpLock> write_locks{};
  unique_ptr<ThrowingOfstream> out_stream;

public:
  ContinuousColorResultsPrinter(
    u64 stream_id_,
    shared_ptr<SharedBatchesProducer<ColorsIntervalBatch>>
      interval_batch_producer_,
    shared_ptr<SharedBatchesProducer<ReadStatisticsBatch>>
      read_statistics_batch_producer_,
    shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
      results_batch_producer_,
    const vector<string> &filenames_,
    u64 num_colors_,
    u64 max_indexes_per_batch,
    u64 max_reads_per_batch,
    double threshold_,
    bool include_not_found_,
    bool include_invalid_,
    u64 threads_,
    u64 warp_size,
    u64 element_size
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
      include_invalid(static_cast<u64>(include_invalid_)),
      threads(threads_),
      write_locks(threads_ - 1),
      stream_id(stream_id_),
      buffers(threads_) {
    for (auto &b : buffers) {
      impl().do_allocate_buffer(
        b,
        max_indexes_per_batch,
        max_reads_per_batch,
        threads,
        warp_size,
        element_size
      );
    }
  }

  auto do_allocate_buffer(
    vector<Buffer_t> &buffer,
    u64 max_indexes_per_batch,
    u64 max_reads_per_batch,
    u64 threads,
    u64 warp_size,
    u64 element_size
  ) -> void {
    buffer.reserve(static_cast<u64>(std::ceil(
      static_cast<double>(max_indexes_per_batch) * num_colors
        / static_cast<double>(threads) / static_cast<double>(warp_size)
        * static_cast<double>(element_size)
      + std::ceil(
        static_cast<double>(max_reads_per_batch) / static_cast<double>(threads)
      )
    )));
  }

  auto read_and_generate() -> void {
    current_filename = filenames.begin();
    if (current_filename == filenames.end()) { return; }
    impl().do_start_next_file();
    for (u64 batch_id = 0; get_batch(); ++batch_id) {
      Logger::log_timed_event(
        format("ResultsPrinter_{}", stream_id),
        Logger::EVENT_STATE::START,
        format("batch {}", batch_id)
      );
      printed_last_read = false;
      process_batch();
      Logger::log_timed_event(
        format("ResultsPrinter_{}", stream_id),
        Logger::EVENT_STATE::STOP,
        format("batch {}", batch_id)
      );
    }
    if (!printed_last_read) {
      u64 buffer_idx = 0;
      buffers[0].resize(num_colors);
      impl().do_print_read(
        previous_last_results.begin(),
        previous_last_found_idx,
        previous_last_not_found_idxs,
        previous_last_invalid_idxs,
        buffers[0],
        buffer_idx
      );
      buffers[0].resize(buffer_idx);
      do_write_buffer(buffers[0], buffer_idx);
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
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

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
    if (wbnrs[0] == 0) {
      u64 buffer_idx = 0;
      buffers[0].resize(num_colors);
      impl().do_print_read(
        previous_last_results.begin(),
        previous_last_found_idx,
        previous_last_not_found_idxs,
        previous_last_invalid_idxs,
        buffers[0],
        buffer_idx
      );
      buffers[0].resize(buffer_idx);
      do_write_buffer(buffers[0], buffer_idx);
      printed_last_read = true;
    } else {
      // Fill in from previous batch (read is continued)
      found_idxs[0] += previous_last_found_idx;
      not_found_idxs[0] += previous_last_not_found_idxs;
      invalid_idxs[0] += previous_last_invalid_idxs;
      std::transform(
        results.begin(),
        copy_advance(results.begin(), num_colors),
        previous_last_results.begin(),
        results.data(),
        std::plus<>()
      );
    }
    u64 start_wbnr_idx = printed_last_read ? 1 : 0;
    for (auto rbnf : rbnfs) {
#pragma omp parallel num_threads(threads)
      {
        u64 thread_idx = omp_get_thread_num();
        if (!write_locks.empty() && thread_idx < buffers.size() - 1) {
          write_locks[thread_idx].set_lock();
        }
#pragma omp barrier

        auto &buffer = buffers[thread_idx];
        buffer.resize(buffer.capacity());
        u64 buffer_idx = 0;

#pragma omp for schedule(static)
        for (u64 wbnr_idx = start_wbnr_idx;
             wbnr_idx < std::min(wbnrs.size(), rbnf);
             ++wbnr_idx) {
          const u64 read_idx = wbnr_idx;
          const u64 wbnr = (wbnr_idx == 0) ? 0 : wbnrs[wbnr_idx - 1];
          if (wbnrs[wbnr_idx] == numeric_limits<u64>::max()) {
            previous_last_found_idx = found_idxs[read_idx];
            previous_last_not_found_idxs = not_found_idxs[read_idx];
            previous_last_invalid_idxs = invalid_idxs[read_idx];
            previous_last_results.insert(
              previous_last_results.begin(),
              std::make_move_iterator(
                copy_advance(results.begin(), wbnr * num_colors)
              ),
              std::make_move_iterator(
                copy_advance(results.begin(), (wbnr + 1) * num_colors)
              )
            );
            printed_last_read = false;
          } else {
            impl().do_print_read(
              copy_advance(results.begin(), wbnr * num_colors),
              found_idxs[read_idx],
              not_found_idxs[read_idx],
              invalid_idxs[read_idx],
              buffer,
              buffer_idx
            );
          }
        }
        buffer.resize(static_cast<std::streamsize>(buffer_idx));
        write_buffers_parallel();
      }
      start_wbnr_idx = std::min(wbnrs.size(), rbnf);
      if (rbnf == start_wbnr_idx) { impl().do_start_next_file(); }
    }
  }

  auto do_print_read(
    vector<u64>::iterator results,
    u64 found_idxs,
    u64 not_found_idxs,
    u64 invalid_idxs,
    vector<Buffer_t> &buffer,
    u64 &buffer_idx
  ) -> void {
    u64 read_size = found_idxs + include_not_found * not_found_idxs
      + include_invalid * invalid_idxs;
    const u64 minimum_found
      = static_cast<u64>(std::ceil(static_cast<double>(read_size) * threshold));
    bool first_print = true;
    for (u64 color_idx = 0; minimum_found > 0 && color_idx < num_colors;
         ++color_idx, ++results) {
      if (*results >= minimum_found) {
        if (!first_print) {
          buffer_idx
            += impl().do_with_space(copy_advance(buffer.begin(), buffer_idx));
        }
        first_print = false;
        buffer_idx += impl().do_with_result(
          copy_advance(buffer.begin(), buffer_idx), color_idx
        );
      }
    }
    buffer_idx
      += impl().do_with_newline(copy_advance(buffer.begin(), buffer_idx));
  }

  auto do_with_newline(vector<Buffer_t>::iterator buffer) -> u64;
  auto do_with_space(vector<Buffer_t>::iterator buffer) -> u64 { return 0; }
  auto do_with_result(vector<Buffer_t>::iterator buffer, u64 result) -> u64;

  auto write_buffers_parallel() -> void {
    u64 thread_idx = omp_get_thread_num();
    auto &buffer = buffers[thread_idx];
    if (thread_idx > 0) { write_locks[thread_idx - 1].set_lock(); }
    impl().do_write_buffer(buffer, buffer.size());
    if (thread_idx > 0) { write_locks[thread_idx - 1].unset_lock(); }
    if (thread_idx < write_locks.size()) {
      write_locks[thread_idx].unset_lock();
    }
  }

  auto do_write_buffer(const vector<Buffer_t> &buffer, u64 amount) -> void {
    out_stream->write(
      bit_cast<char *>(buffer.data()),
      static_cast<std::streamsize>(amount * sizeof(Buffer_t))
    );
  }
};

}  // namespace sbwt_search

#endif
