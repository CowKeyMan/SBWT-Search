#ifndef COLOR_SEARCHER_H
#define COLOR_SEARCHER_H

/**
 * @file ColorSearcher.h
 * @brief Offloads color searching to the gpu, given sbwt indexes
 */

#include <memory>

#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "Tools/GpuEvent.h"
#include "Tools/GpuStream.h"

namespace sbwt_search {

using gpu_utils::GpuEvent;
using gpu_utils::GpuStream;
using std::shared_ptr;

class ColorSearcher {
private:
  GpuStream gpu_stream{};
  shared_ptr<GpuColorIndexContainer> container;
  GpuPointer<u64> d_sbwt_index_idxs;
  GpuPointer<u8> d_fat_results;
  GpuPointer<u64> d_results;
  GpuEvent start_timer{}, end_timer{};
  u64 stream_id;

public:
  ColorSearcher(
    u64 stream_id_,
    shared_ptr<GpuColorIndexContainer> container,
    u64 max_indexes_per_batch,
    u64 max_seqs_per_batch
  );

  auto search(
    const vector<u64> &sbwt_index_idxs,
    const vector<u64> &warps_intervals,
    vector<u64> &results,
    u64 batch_id
  ) -> void;

private:
  auto searcher_copy_to_gpu(u64 batch_id, const vector<u64> &sbwt_index_ids)
    -> void;
  auto launch_search_kernel(u64 num_queries, u64 batch_id) -> void;

  auto
  combine_copy_to_gpu(u64 batch_id, const vector<u64> &warps_before_new_read)
    -> void;
  auto launch_combine_kernel(u64 num_warps, u64 num_colors, u64 batch_id)
    -> void;
  auto copy_from_gpu(vector<u64> &results, u64 batch_id) -> void;
  auto fill_unique_warps_before_new_read(
    const vector<u64> &warps_before_new_read, u64 num_warps
  ) -> void;
};

}  // namespace sbwt_search

#endif
