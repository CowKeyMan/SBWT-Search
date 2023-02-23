#ifndef COLOR_SEARCHER_H
#define COLOR_SEARCHER_H

/**
 * @file ColorSearcher.h
 * @brief Offloads color searching to the gpu, given sbwt indexes
 */

#include <memory>

#include "ColorIndexContainer/GpuColorIndexContainer.h"

namespace sbwt_search {

using std::shared_ptr;

class ColorSearcher {
private:
  shared_ptr<GpuColorIndexContainer> container;
  GpuPointer<u64> d_sbwt_index_idxs;
  GpuPointer<u64> d_results;

public:
  ColorSearcher(
    shared_ptr<GpuColorIndexContainer> container, u64 max_indexes_per_batch
  );

  auto
  search(const vector<u64> &sbwt_index_idxs, vector<u64> &results, u64 batch_id)
    -> void;

private:
  auto copy_to_gpu(
    u64 batch_id, const vector<u64> &sbwt_index_ids, vector<u64> &results
  ) -> void;
  auto launch_search_kernel(u64 num_queries, u64 batch_id) -> void;
  auto copy_from_gpu(vector<u64> &results, u64 batch_id) -> void;
};

}  // namespace sbwt_search

#endif
