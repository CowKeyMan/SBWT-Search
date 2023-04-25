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
#include <ostream>

#include "BatchObjects/ColorsBatch.h"
#include "BatchObjects/SeqStatisticsBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/OmpLock.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/StdUtils.hpp"
#include "fmt/core.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using fmt::format;
using io_utils::ThrowingOfstream;
using log_utils::Logger;
using math_utils::divide_and_ceil;
using std::bit_cast;
using std::ios;
using std::numeric_limits;
using std::ostream;
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
  shared_ptr<SharedBatchesProducer<SeqStatisticsBatch>>
    seq_statistics_batch_producer;
  shared_ptr<SharedBatchesProducer<ColorsBatch>> colors_batch_producer;
  shared_ptr<SeqStatisticsBatch> seq_statistics_batch;
  shared_ptr<ColorsBatch> colors_batch;
  vector<string> filenames;
  vector<string>::iterator current_filename;
  u64 num_colors;
  double threshold;
  vector<u64> previous_last_results;
  u64 previous_last_found_idx = numeric_limits<u64>::max();
  u64 previous_last_not_found_idxs = numeric_limits<u64>::max();
  u64 previous_last_invalid_idxs = numeric_limits<u64>::max();
  u64 include_not_found;
  u64 include_invalid;
  vector<vector<Buffer_t>> buffers;
  u64 threads;
  u64 stream_id;
  vector<OmpLock> write_locks{};
  unique_ptr<ThrowingOfstream> out_stream;
  bool write_headers;

public:
  ContinuousColorResultsPrinter(
    u64 stream_id_,
    shared_ptr<SharedBatchesProducer<SeqStatisticsBatch>>
      seq_statistics_batch_producer_,
    shared_ptr<SharedBatchesProducer<ColorsBatch>> colors_batch_producer_,
    const vector<string> &filenames_,
    u64 num_colors_,
    double threshold_,
    bool include_not_found_,
    bool include_invalid_,
    u64 threads_,
    u64 read_size,
    u64 max_reads_per_batch,
    bool write_headers_
  ):
      seq_statistics_batch_producer(std::move(seq_statistics_batch_producer_)),
      colors_batch_producer(std::move(colors_batch_producer_)),
      filenames(filenames_),
      num_colors(num_colors_),
      threshold(threshold_),
      previous_last_results(num_colors_, 0),
      include_not_found(static_cast<u64>(include_not_found_)),
      include_invalid(static_cast<u64>(include_invalid_)),
      threads(threads_),
      write_locks(threads_ - 1),
      stream_id(stream_id_),
      buffers(threads_),
      write_headers(write_headers_) {
    for (auto &b : buffers) {
      impl().do_allocate_buffer(
        b, divide_and_ceil<u64>(max_reads_per_batch, threads_) * read_size
      );
    }
  }

  auto do_allocate_buffer(vector<Buffer_t> &buffer, u64 amount) -> void {
    buffer.resize(amount);
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
      process_batch();
      Logger::log_timed_event(
        format("ResultsPrinter_{}", stream_id),
        Logger::EVENT_STATE::STOP,
        format("batch {}", batch_id)
      );
    }
    impl().do_at_file_end();
  }

private:
  auto get_batch() -> bool {
    return (static_cast<u64>(
              *seq_statistics_batch_producer >> seq_statistics_batch
            )
            & static_cast<u64>(*colors_batch_producer >> colors_batch))
      > 0;
  }

protected:
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

  auto do_start_next_file() -> void {
    if (current_filename != filenames.begin()) { impl().do_at_file_end(); }
    impl().do_open_next_file(*current_filename);
    impl().do_write_file_header(*out_stream);
    current_filename = next(current_filename);
  }
  auto do_at_file_end() -> void {}
  auto do_open_next_file(const string &filename) -> void {
    out_stream = make_unique<ThrowingOfstream>(
      filename + impl().do_get_extension(), ios::binary | ios::out
    );
  }
  auto do_write_file_header(ThrowingOfstream &out_stream) -> void {
    if (write_headers) {
      out_stream.write_string_with_size(impl().do_get_format());
      out_stream.write_string_with_size(impl().do_get_version());
    }
  }

  auto process_batch() -> void {
    auto &colors = colors_batch->colors;
    auto &found_idxs = seq_statistics_batch->found_idxs;
    auto &not_found_idxs = seq_statistics_batch->not_found_idxs;
    auto &invalid_idxs = seq_statistics_batch->invalid_idxs;
    auto &colored_seq_id = seq_statistics_batch->colored_seq_id;
    auto &sbnfs = seq_statistics_batch->seqs_before_new_file;

    // Fill in from previous batch (read is continued)
    found_idxs[0] += previous_last_found_idx;
    not_found_idxs[0] += previous_last_not_found_idxs;
    invalid_idxs[0] += previous_last_invalid_idxs;
    std::transform(
      colors.begin(),
      copy_advance(colors.begin(), num_colors),
      previous_last_results.begin(),
      colors.data(),
      std::plus<>()
    );
    u64 start_seq = 0;
    for (u64 sbnf_idx = 0; sbnf_idx < sbnfs.size(); ++sbnf_idx) {
      u64 end_seq = std::min(sbnfs[sbnf_idx], colored_seq_id.size() - 1);
#pragma omp parallel num_threads(threads)
      {
        u64 thread_idx = omp_get_thread_num();
        if (!write_locks.empty() && thread_idx < buffers.size() - 1) {
          write_locks[thread_idx].set_lock();
        }
#pragma omp barrier

        auto &buffer = buffers[thread_idx];
        u64 buffer_idx = 0;

#pragma omp for schedule(static)
        for (u64 seq_idx = start_seq; seq_idx < end_seq; ++seq_idx) {
          impl().do_print_read(
            copy_advance(colors.begin(), colored_seq_id[seq_idx] * num_colors),
            found_idxs[seq_idx],
            not_found_idxs[seq_idx],
            invalid_idxs[seq_idx],
            buffer,
            buffer_idx
          );
        }
        write_buffers_parallel(buffer_idx);
      }
      if (end_seq == sbnfs[sbnf_idx]) { impl().do_start_next_file(); }
      start_seq = end_seq;
    }

    previous_last_found_idx = found_idxs.back();
    previous_last_not_found_idxs = not_found_idxs.back();
    previous_last_invalid_idxs = invalid_idxs.back();
    if (previous_last_found_idx > 0) {
      previous_last_results.insert(
        previous_last_results.begin(),
        std::make_move_iterator(
          copy_advance(colors.begin(), colored_seq_id.back() * num_colors)
        ),
        std::make_move_iterator(
          copy_advance(colors.begin(), (colored_seq_id.back() + 1) * num_colors)
        )
      );
    } else {
      previous_last_results.assign(previous_last_results.size(), 0);
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

  auto write_buffers_parallel(u64 buffer_idx) -> void {
    u64 thread_idx = omp_get_thread_num();
    auto &buffer = buffers[thread_idx];
    if (thread_idx > 0) { write_locks[thread_idx - 1].set_lock(); }
    impl().do_write_buffer(buffer, buffer_idx);
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
