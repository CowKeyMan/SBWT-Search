#include <limits>

#include "ColorSearcher/ColorSearcher.h"
#include "Global/GlobalDefinitions.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using math_utils::round_up;
using std::numeric_limits;

ColorSearcher::ColorSearcher(
  u64 stream_id_,
  shared_ptr<GpuColorIndexContainer> container_,
  u64 max_indexes_per_batch
):
    container(std::move(container_)),
    d_sbwt_index_idxs(max_indexes_per_batch),
    d_fat_results(
      max_indexes_per_batch / gpu_warp_size * container->num_colors, gpu_stream
    ),
    d_results(
      max_indexes_per_batch / gpu_warp_size * container->num_colors, gpu_stream
    ),
    stream_id(stream_id_) {}

auto ColorSearcher::search(
  const vector<u64> &sbwt_index_idxs,
  const vector<u64> &warps_before_new_read,
  vector<u64> &results,
  u64 batch_id
) -> void {
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} consists of {} queries", batch_id, sbwt_index_idxs.size())
  );
  if (!sbwt_index_idxs.empty()) {
    searcher_copy_to_gpu(batch_id, sbwt_index_idxs);
    results.resize(warps_before_new_read.size() * container->num_colors);
    launch_search_kernel(sbwt_index_idxs.size(), batch_id);
    combine_copy_to_gpu(batch_id, warps_before_new_read);
    launch_combine_kernel(
      sbwt_index_idxs.size() / gpu_warp_size,
      warps_before_new_read.size(),
      container->num_colors,
      batch_id
    );
    copy_from_gpu(results, batch_id);
  }
}

auto ColorSearcher::searcher_copy_to_gpu(
  u64 batch_id, const vector<u64> &sbwt_index_idxs
) -> void {
  Logger::log_timed_event(
    format("SearcherCopyToGpu1_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  auto padded_query_size
    = round_up<u64>(sbwt_index_idxs.size(), superblock_bits);
  d_sbwt_index_idxs.set_async(
    sbwt_index_idxs, sbwt_index_idxs.size(), gpu_stream
  );
  d_sbwt_index_idxs.memset_async(
    sbwt_index_idxs.size(),
    padded_query_size - sbwt_index_idxs.size(),
    numeric_limits<u8>::max(),
    gpu_stream
  );
  Logger::log_timed_event(
    format("SearcherCopyToGpu1_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

auto ColorSearcher::combine_copy_to_gpu(
  u64 batch_id, const vector<u64> &warps_before_new_read
) -> void {
  Logger::log_timed_event(
    format("SearcherCopyToGpu2_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  auto &d_warps_before_new_read = d_sbwt_index_idxs;
  d_warps_before_new_read.set_async(
    warps_before_new_read.data(), warps_before_new_read.size(), gpu_stream
  );
  Logger::log_timed_event(
    format("SearcherCopyToGpu2_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

auto ColorSearcher::copy_from_gpu(vector<u64> &results, u64 batch_id) -> void {
  Logger::log_timed_event(
    format("SearcherCopyFromGpu_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  d_results.copy_to_async(results, results.size(), gpu_stream);
  Logger::log_timed_event(
    format("SearcherCopyFromGpu_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

}  // namespace sbwt_search
