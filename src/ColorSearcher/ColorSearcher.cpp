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
  shared_ptr<GpuColorIndexContainer> container_, u64 max_indexes_per_batch
):
    container(std::move(container_)),
    d_sbwt_index_idxs(max_indexes_per_batch),
    d_results(10000) {}  // TODO: resize results appropriately

auto ColorSearcher::search(
  const vector<u64> &sbwt_index_idxs, vector<u64> &results, u64 batch_id
) -> void {
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} consists of {} queries", batch_id, results.size())
  );
  copy_to_gpu(batch_id, sbwt_index_idxs, results);
  if (!sbwt_index_idxs.empty()) {
    Logger::log_timed_event(
      "SearcherSearch", Logger::EVENT_STATE::START, format("batch {}", batch_id)
    );
    launch_search_kernel(sbwt_index_idxs.size(), batch_id);
    Logger::log_timed_event(
      "SearcherSearch", Logger::EVENT_STATE::STOP, format("batch {}", batch_id)
    );
    copy_from_gpu(results, batch_id);
  }
}

auto ColorSearcher::copy_to_gpu(
  u64 batch_id, const vector<u64> &sbwt_index_idxs, vector<u64> &results
) -> void {
  Logger::log_timed_event(
    "SearcherCopyToGpu",
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  auto padded_query_size
    = round_up<u64>(sbwt_index_idxs.size(), superblock_bits);
  d_sbwt_index_idxs.set(sbwt_index_idxs, sbwt_index_idxs.size());
  d_sbwt_index_idxs.memset(
    sbwt_index_idxs.size(),
    padded_query_size - sbwt_index_idxs.size(),
    numeric_limits<u8>::max()
  );
  Logger::log_timed_event(
    "SearcherCopyToGpu", Logger::EVENT_STATE::STOP, format("batch {}", batch_id)
  );
  // results.resize(kmer_positions.size()); // TODO: resize properly
}

auto ColorSearcher::copy_from_gpu(vector<u64> &results, u64 batch_id) -> void {
  Logger::log_timed_event(
    "SearcherCopyFromGpu",
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  d_results.copy_to(results);
  Logger::log_timed_event(
    "SearcherCopyFromGpu",
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

}  // namespace sbwt_search
