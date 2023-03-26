#ifndef INDEX_SEARCHER_H
#define INDEX_SEARCHER_H

/**
 * @file IndexSearcher.h
 * @brief Class for searching the SBWT index
 */

#include <memory>

#include "SbwtContainer/GpuSbwtContainer.h"
#include "Tools/GpuEvent.h"
#include "Tools/GpuStream.h"

namespace sbwt_search {

using gpu_utils::GpuEvent;
using gpu_utils::GpuStream;
using std::shared_ptr;

class IndexSearcher {
private:
  shared_ptr<GpuSbwtContainer> container;
  GpuPointer<u64> d_bit_seqs;
  GpuPointer<u64> d_kmer_positions;
  GpuEvent start_timer{}, end_timer{};
  GpuStream gpu_stream;
  u64 stream_id;

public:
  IndexSearcher(
    u64 stream_id_,
    shared_ptr<GpuSbwtContainer> container,
    u64 max_chars_per_batch
  );

  auto search(
    const vector<u64> &bit_seqs,
    const vector<u64> &kmer_positions,
    vector<u64> &results,
    u64 batch_id
  ) -> void;

private:
  auto copy_to_gpu(
    u64 batch_id,
    const vector<u64> &bit_seqs,
    const vector<u64> &kmer_positions,
    vector<u64> &results
  ) -> void;
  auto launch_search_kernel(u64 num_queries, u64 batch_id) -> void;
  auto copy_from_gpu(
    vector<u64> &results, u64 batch_id, const vector<u64> &kmer_positions
  ) -> void;
};

}  // namespace sbwt_search

#endif
