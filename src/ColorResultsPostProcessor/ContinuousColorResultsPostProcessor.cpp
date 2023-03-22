#include <cmath>
#include <memory>
#include <omp.h>

#include "ColorResultsPostProcessor/ContinuousColorResultsPostProcessor.h"
#include "Tools/Logger.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using std::make_shared;

ContinuousColorResultsPostProcessor::ContinuousColorResultsPostProcessor(
  u64 stream_id_,
  shared_ptr<SharedBatchesProducer<ColorSearchResultsBatch>>
    results_batch_producer_,
  shared_ptr<SharedBatchesProducer<WarpsBeforeNewReadBatch>>
    warps_before_new_read_batch_producer_,
  u64 max_batches,
  u64 num_colors_
):
    SharedBatchesProducer<ColorSearchResultsBatch>(max_batches),
    results_batch_producer(std::move(results_batch_producer_)),
    warps_before_new_read_batch_producer(
      std::move(warps_before_new_read_batch_producer_)
    ),
    num_colors(num_colors_),
    stream_id(stream_id_) {
  initialise_batches();
}

auto ContinuousColorResultsPostProcessor::continue_read_condition() -> bool {
  return static_cast<bool>(
    static_cast<u64>(*results_batch_producer >> results_batch)
    & static_cast<u64>(
      *warps_before_new_read_batch_producer >> warps_before_new_read_batch
    )
  );
}

auto ContinuousColorResultsPostProcessor::generate() -> void {
  squeeze_results();
  current_write()->results = results_batch->results;
}

auto ContinuousColorResultsPostProcessor::squeeze_results() -> void {
#pragma omp parallel
  {
    u64 thread_idx = omp_get_thread_num();
    u64 start_color_idx = static_cast<u64>(std::round(
      static_cast<double>(thread_idx) * static_cast<double>(num_colors)
      / omp_get_num_threads()
    ));
    u64 end_color_idx = static_cast<u64>(std::round(
      static_cast<double>(thread_idx + 1) * static_cast<double>(num_colors)
      / omp_get_num_threads()
    ));
    u64 start_warp = 0;
    for (auto end_warp : *warps_before_new_read_batch->warps_before_new_read) {
      end_warp
        = std::min(end_warp, results_batch->results->size() / num_colors);
      if (start_warp == end_warp) { continue; }
#pragma omp simd
      for (u64 c = start_color_idx; c < end_color_idx; ++c) {
        u64 result = 0;
        for (u64 warp = start_warp; warp < end_warp; ++warp) {
          result += (*results_batch->results)[warp * num_colors + c];
        }
        (*results_batch->results)[start_warp * num_colors + c] = result;
      }
      start_warp = end_warp;
    }
  }
}

auto ContinuousColorResultsPostProcessor::get_default_value()
  -> shared_ptr<ColorSearchResultsBatch> {
  return make_shared<ColorSearchResultsBatch>();
}

auto ContinuousColorResultsPostProcessor::do_at_batch_start() -> void {
  SharedBatchesProducer<ColorSearchResultsBatch>::do_at_batch_start();
  Logger::log_timed_event(
    format("ResultsPostProcessor_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", get_batch_id())
  );
}

auto ContinuousColorResultsPostProcessor::do_at_batch_finish() -> void {
  Logger::log_timed_event(
    format("ResultsPostProcessor_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", get_batch_id())
  );
  SharedBatchesProducer<ColorSearchResultsBatch>::do_at_batch_finish();
}

}  // namespace sbwt_search
