#include "Global/GlobalDefinitions.h"
#include "IndexSearcher/IndexSearcher.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using math_utils::round_up;

IndexSearcher::IndexSearcher(
  u64 stream_id_,
  shared_ptr<GpuSbwtContainer> container,
  u64 max_chars_per_batch,
  bool move_to_key_kmer_
):
    container(std::move(container)),
    d_bit_seqs(max_chars_per_batch / u64_bits * 2, gpu_stream),
    d_kmer_positions(max_chars_per_batch, gpu_stream),
    stream_id(stream_id_),
    move_to_key_kmer(move_to_key_kmer_) {}

auto IndexSearcher::search(
  const PinnedVector<u64> &bit_seqs,
  const PinnedVector<u64> &kmer_positions,
  PinnedVector<u64> &results,
  u64 batch_id
) -> void {
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} consists of {} queries", batch_id, kmer_positions.size())
  );
  copy_to_gpu(batch_id, bit_seqs, kmer_positions, results);
  if (!kmer_positions.empty()) {
    Logger::log_timed_event(
      format("SearcherSearch_{}", stream_id),
      Logger::EVENT_STATE::START,
      format("batch {}", batch_id)
    );
    launch_search_kernel(kmer_positions.size(), batch_id);
    Logger::log_timed_event(
      format("SearcherSearch_{}", stream_id),
      Logger::EVENT_STATE::STOP,
      format("batch {}", batch_id)
    );
    copy_from_gpu(results, batch_id, kmer_positions);
  }
}

auto IndexSearcher::copy_to_gpu(
  u64 batch_id,
  const PinnedVector<u64> &bit_seqs,
  const PinnedVector<u64> &kmer_positions,
  PinnedVector<u64> &results
) -> void {
  Logger::log_timed_event(
    format("SearcherCopyToGpu_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  d_bit_seqs.set_async(bit_seqs.data(), bit_seqs.size(), gpu_stream);
  auto padded_query_size
    = round_up<u64>(kmer_positions.size(), superblock_bits);
  d_kmer_positions.set_async(
    kmer_positions.data(), kmer_positions.size(), gpu_stream
  );
  d_kmer_positions.memset_async(
    kmer_positions.size(),
    padded_query_size - kmer_positions.size(),
    0,
    gpu_stream
  );
  Logger::log_timed_event(
    format("SearcherCopyToGpu_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
  results.resize(kmer_positions.size());
}

auto IndexSearcher::copy_from_gpu(
  PinnedVector<u64> &results,
  u64 batch_id,
  const PinnedVector<u64> &kmer_positions
) -> void {
  Logger::log_timed_event(
    format("SearcherCopyFromGpu_{}", stream_id),
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  d_kmer_positions.copy_to_async(
    results.data(), kmer_positions.size(), gpu_stream
  );
  Logger::log_timed_event(
    format("SearcherCopyFromGpu_{}", stream_id),
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

}  // namespace sbwt_search
