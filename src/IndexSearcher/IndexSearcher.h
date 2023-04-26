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
#include "Tools/PinnedVector.h"

namespace sbwt_search {

using gpu_utils::GpuEvent;
using gpu_utils::GpuStream;
using gpu_utils::PinnedVector;
using std::shared_ptr;

class IndexSearcher {
private:
  GpuStream gpu_stream{};
  shared_ptr<GpuSbwtContainer> container;
  GpuPointer<u64> d_bit_seqs;
  GpuPointer<u64> d_kmer_positions;
  GpuEvent start_timer{}, end_timer{};
  u64 stream_id;
  bool move_to_key_kmer;

public:
  IndexSearcher(
    u64 stream_id_,
    shared_ptr<GpuSbwtContainer> container,
    u64 max_chars_per_batch,
    bool move_to_key_kmer_
  );

  auto search(
    const PinnedVector<u64> &bit_seqs,
    const PinnedVector<u64> &kmer_positions,
    PinnedVector<u64> &results,
    u64 batch_id
  ) -> void;

private:
  auto copy_to_gpu(
    u64 batch_id,
    const PinnedVector<u64> &bit_seqs,
    const PinnedVector<u64> &kmer_positions,
    PinnedVector<u64> &results
  ) -> void;
  auto launch_search_kernel(u64 num_queries, u64 batch_id) -> void;
  auto copy_from_gpu(
    PinnedVector<u64> &results,
    u64 batch_id,
    const PinnedVector<u64> &kmer_positions
  ) -> void;
};

}  // namespace sbwt_search

#endif
