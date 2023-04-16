#ifndef CONTINUOUS_INDEX_RESULTS_PRINTER_HPP
#define CONTINUOUS_INDEX_RESULTS_PRINTER_HPP

/**
 * @file ContinuousIndexResultsPrinter.hpp
 * @brief Gets results, intervals and list of invalid chars and prints
 * these out to disk based on the given data and filenames. This printing is
 * done in parallel. This class uses the Template Pattern, with CRTP. Honestly
 * its a disgusting class, mosly due to all the variables it needs to manage in
 * order to be efficient and highly parallel. All the variable names are also
 * super long, and then I tried to give them some short name but it still ended
 * up seeming obscure. I tried my best to make it as easy to understand as
 * possible, but good luck!
 */

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>

#include "BatchObjects/IntervalBatch.h"
#include "BatchObjects/InvalidCharsBatch.h"
#include "BatchObjects/ResultsBatch.h"
#include "BatchObjects/StringSequenceBatch.h"
#include "Tools/IOUtils.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/OmpLock.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/StdUtils.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"

namespace sbwt_search {

using design_utils::SharedBatchesProducer;
using fmt::format;
using io_utils::ThrowingOfstream;
using log_utils::Logger;
using math_utils::divide_and_ceil;
using math_utils::round_up;
using std::bit_cast;
using std::ceil;
using std::ios;
using std::make_unique;
using std::min;
using std::next;
using std::numeric_limits;
using std::shared_ptr;
using std::unique_ptr;
using std_utils::copy_advance;
using threading_utils::OmpLock;

template <class TImplementation, class Buffer_t>
class ContinuousIndexResultsPrinter {
private:
  vector<u64> results_before_newline{};
  vector<OmpLock> write_locks{};
  auto impl() -> TImplementation & {
    return static_cast<TImplementation &>(*this);
  }
  shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer;
  shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer;
  shared_ptr<SharedBatchesProducer<InvalidCharsBatch>> invalid_chars_producer;
  vector<string> filenames;
  vector<string>::iterator current_filename;
  shared_ptr<ResultsBatch> results_batch;
  shared_ptr<InvalidCharsBatch> invalid_chars_batch;
  shared_ptr<IntervalBatch> interval_batch;
  unique_ptr<ThrowingOfstream> out_stream;
  vector<vector<Buffer_t>> buffers;
  u64 kmer_size;
  u64 threads;
  u64 stream_id;
  bool write_headers;

public:
  ContinuousIndexResultsPrinter(
    u64 stream_id_,
    shared_ptr<SharedBatchesProducer<ResultsBatch>> results_producer_,
    shared_ptr<SharedBatchesProducer<IntervalBatch>> interval_producer_,
    shared_ptr<SharedBatchesProducer<InvalidCharsBatch>>
      invalid_chars_producer_,
    vector<string> filenames_,
    u64 kmer_size,
    u64 threads_,
    u64 max_chars_per_batch,
    u64 max_reads_per_batch,
    u64 element_size,
    u64 newline_element_size,
    bool write_headers_
  ):
      results_producer(std::move(results_producer_)),
      interval_producer(std::move(interval_producer_)),
      invalid_chars_producer(std::move(invalid_chars_producer_)),
      filenames(std::move(filenames_)),
      threads(threads_),
      kmer_size(kmer_size),
      write_locks(threads_ - 1),
      stream_id(stream_id_),
      buffers(threads_),
      write_headers(write_headers_) {
    for (auto &b : buffers) {
      impl().do_allocate_buffer(
        b,
        max_chars_per_batch,
        max_reads_per_batch,
        threads,
        element_size,
        newline_element_size
      );
    }
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
      impl().process_batch();
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
    return (static_cast<u64>(*interval_producer >> interval_batch)
            & static_cast<u64>(*invalid_chars_producer >> invalid_chars_batch)
            & static_cast<u64>(*results_producer >> results_batch))
      > 0;
  }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  auto process_batch() -> void {
    populate_results_before_newline();
    const auto &results = results_batch->results;
    const auto &invalid_chars = invalid_chars_batch->invalid_chars;
    const auto &nlbnfs = interval_batch->newlines_before_newfile;
    const auto &cbnls = *interval_batch->chars_before_newline;
    const auto &rbnls = results_before_newline;
    u64 prev_last_results_idx = 0;
    // for each file
    for (u64 nlbnf_idx = 0; nlbnf_idx < nlbnfs.size(); ++nlbnf_idx) {
      u64 nlbnf = nlbnfs[nlbnf_idx];
      u64 first_results_idx = prev_last_results_idx;
      u64 last_results_idx = results.size();
      if (nlbnf < numeric_limits<u64>::max()) {
        last_results_idx = rbnls[nlbnf - 1];
      }
      u64 results_in_file = last_results_idx - first_results_idx;
      u64 rbnl_idx = nlbnf_idx > 0 ? nlbnfs[nlbnf_idx - 1] : 0;
      dump_starting_newlines(first_results_idx, rbnl_idx, nlbnf_idx);
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
        u64 start_idx = static_cast<u64>(round(
          (static_cast<double>(results_in_file)
           / static_cast<double>(buffers.size()))
          * static_cast<double>(thread_idx)
        ));
        u64 end_idx = min(
          results_in_file,
          static_cast<u64>(round(
            (static_cast<double>(results_in_file)
             / static_cast<double>(buffers.size()))
            * static_cast<double>(thread_idx + 1)
          ))
        );
        u64 result_idx = first_results_idx + start_idx;
        u64 bnl_idx = std::upper_bound(rbnls.begin(), rbnls.end(), result_idx)
          - rbnls.begin();
        u64 char_idx = static_cast<u64>(bnl_idx > 0)
            * (cbnls[bnl_idx - 1] - rbnls[bnl_idx - 1])
          + first_results_idx + start_idx;

        u64 invalid_chars_left
          = get_invalid_chars_left_first_kmer(char_idx, cbnls[bnl_idx]);

        for (u64 i = start_idx; i < end_idx; ++i, ++result_idx) {
          if (invalid_chars[char_idx + kmer_size - 1] == 1) {
            invalid_chars_left = kmer_size;
          }
          add_new_result(
            buffer, invalid_chars_left, buffer_idx, results[result_idx]
          );
          bool newline = false;
          while (result_idx + 1 == rbnls[bnl_idx] && bnl_idx < nlbnf) {
            newline = true;
            buffer_idx
              += impl().do_with_newline(copy_advance(buffer.begin(), buffer_idx)
              );
            ++bnl_idx;
          }
          if (newline) {
            char_idx = cbnls[bnl_idx - 1];
            invalid_chars_left
              = get_invalid_chars_left_first_kmer(char_idx, cbnls[bnl_idx]);
          } else {
            ++char_idx;
            buffer_idx
              += impl().do_with_space(copy_advance(buffer.begin(), buffer_idx));
          }
        }
        buffer.resize(static_cast<std::streamsize>(buffer_idx));
        write_buffers_parallel();
      }
      impl().do_at_file_end();
      prev_last_results_idx = last_results_idx;
      if (nlbnf_idx + 1 < nlbnfs.size()) { do_start_next_file(); }
    }
  }

  auto populate_results_before_newline() -> void {
    const auto &cbnl = *interval_batch->chars_before_newline;
    auto &rbnl = results_before_newline;
    rbnl.resize(cbnl.size());
    u64 last_rbnl = 0;
    u64 last_cbnl = 0;
    for (int i = 0; i < cbnl.size() - 1; ++i) {
      if ((cbnl[i] - last_cbnl) < kmer_size) {
        rbnl[i] = last_rbnl;
      } else {
        rbnl[i] = last_rbnl + cbnl[i] - last_cbnl - (kmer_size - 1);
      }
      last_rbnl = rbnl[i];
      last_cbnl = cbnl[i];
    }
    rbnl.back() = cbnl.back();
  }

  auto dump_starting_newlines(u64 res_idx, u64 &rbnl_idx, u64 nlbnf_idx)
    -> void {
    u64 buffer_idx = 0;
    auto &buffer = buffers[0];
    buffer.resize(buffer.capacity());
    const auto &nlbnfs = interval_batch->newlines_before_newfile;
    const auto &rbnls = results_before_newline;
    while (rbnls[rbnl_idx] == res_idx && rbnl_idx < nlbnfs[nlbnf_idx]) {
      buffer_idx
        += impl().do_with_newline(copy_advance(buffer.begin(), buffer_idx));
      ++rbnl_idx;
    }
    impl().do_write_buffer(buffer, buffer_idx);
  }

  auto get_invalid_chars_left_first_kmer(u64 char_idx, u64 chars_before_newline)
    -> u64 {
    auto &invalid_chars = invalid_chars_batch->invalid_chars;
    auto limit
      = min({char_idx + kmer_size, invalid_chars.size(), chars_before_newline});
    if (limit <= char_idx) { return 0; }
    for (u64 i = limit; i > char_idx; --i) {
      if (invalid_chars[i - 1] == 1) { return i - char_idx; }
    }
    return 0;
  }

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

  auto add_new_result(
    vector<Buffer_t> &buffer,
    u64 &invalid_chars_left,
    u64 &buffer_idx,
    u64 result
  ) -> void {
    const auto &results = results_batch->results;
    if (invalid_chars_left > 0) {
      buffer_idx
        += impl().do_with_invalid(copy_advance(buffer.begin(), buffer_idx));
      --invalid_chars_left;
    } else if (result == numeric_limits<u64>::max()) {
      buffer_idx
        += impl().do_with_not_found(copy_advance(buffer.begin(), buffer_idx));
    } else {
      buffer_idx += impl().do_with_result(
        copy_advance(buffer.begin(), buffer_idx), result
      );
    }
  }

protected:
  auto do_get_extension() -> string;
  auto do_get_format() -> string;
  auto do_get_version() -> string;

  auto do_allocate_buffer(
    vector<Buffer_t> &buffer,
    u64 max_chars_per_batch,
    u64 max_reads_per_batch,
    u64 threads,
    u64 element_size,
    u64 newline_element_size
  ) -> void {
    buffer.reserve(
      divide_and_ceil<u64>(max_chars_per_batch, threads) * element_size
      + divide_and_ceil<u64>(max_reads_per_batch, threads)
        * newline_element_size
    );
  }

  auto do_write_buffer(const vector<Buffer_t> &buffer, u64 amount) -> void {
    out_stream->write(
      bit_cast<char *>(buffer.data()),
      static_cast<std::streamsize>(amount * sizeof(Buffer_t))
    );
  }

  auto do_write_file_header() -> void {
    out_stream->write_string_with_size(impl().do_get_format());
    out_stream->write_string_with_size(impl().do_get_version());
  }

  auto do_start_next_file() -> void {
    if (current_filename != filenames.begin()) { impl().do_at_file_end(); }
    impl().do_open_next_file(*current_filename);
    if (this->write_headers) { impl().do_write_file_header(); }
    current_filename = next(current_filename);
  }

  auto do_open_next_file(const string &filename) -> void {
    out_stream = make_unique<ThrowingOfstream>(
      filename + impl().do_get_extension(), ios::binary | ios::out
    );
  };

  auto do_with_result(vector<Buffer_t>::iterator buffer, u64 result) -> u64;
  auto do_with_invalid(vector<Buffer_t>::iterator buffer) -> u64;
  auto do_with_not_found(vector<Buffer_t>::iterator buffer) -> u64;
  auto do_with_newline(vector<Buffer_t>::iterator buffer) -> u64;
  auto do_with_space(vector<Buffer_t>::iterator buffer) -> u64 { return 0; }

  auto do_at_file_end() -> void{};
};

}  // namespace sbwt_search

#endif
