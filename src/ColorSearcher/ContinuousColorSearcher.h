#ifndef CONTINUOUS_COLOR_SEARCHER_H
#define CONTINUOUS_COLOR_SEARCHER_H

/**
 * @file ContinuousColorSearcher.h
 * @brief
 */

#include <memory>

#include "BatchObjects/ColorsBatch.h"
#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "ColorSearcher/ColorSearcher.h"
#include "IndexFileParser/IndexesBatchProducer.h"
#include "Tools/SharedBatchesProducer.hpp"

namespace sbwt_search {

using std::shared_ptr;

using design_utils::SharedBatchesProducer;

class ContinuousColorSearcher: public SharedBatchesProducer<ColorsBatch> {
private:
  shared_ptr<SharedBatchesProducer<IndexesBatch>> indexes_batch_producer;
  shared_ptr<IndexesBatch> indexes_batch;
  ColorSearcher searcher;
  u64 max_indexes_per_batch;
  u64 max_seqs_per_batch;
  u64 num_colors;
  u64 stream_id;

public:
  ContinuousColorSearcher(
    u64 stream_id_,
    shared_ptr<GpuColorIndexContainer> color_index_container_,
    shared_ptr<SharedBatchesProducer<IndexesBatch>> indexes_batch_producer_,
    u64 max_indexes_per_batch_,
    u64 max_seqs_per_batch_,
    u64 max_batches,
    u64 num_colors_
  );

  auto static get_bits_per_seq_cpu(u64 num_colors) -> u64;
  auto static get_bits_per_warp_gpu(u64 num_colors) -> u64;

  auto static get_bits_per_element_gpu(u64 num_colors, u64 idxs_per_seq)
    -> double;

private:
  auto get_default_value() -> shared_ptr<ColorsBatch> override;
  auto start_new_batch() -> void;
  auto continue_read_condition() -> bool override;
  auto generate() -> void override;
  auto do_at_batch_start() -> void override;
  auto do_at_batch_finish() -> void override;
};

}  // namespace sbwt_search

#endif
