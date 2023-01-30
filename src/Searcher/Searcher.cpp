#include "Global/GlobalDefinitions.h"
#include "Searcher/Searcher.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/TypeDefinitions.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using math_utils::round_up;

Searcher::Searcher(
  shared_ptr<GpuSbwtContainer> container, u64 max_chars_per_batch
):
  container(std::move(container)),
  d_bit_seqs(max_chars_per_batch / u64_bits * 2),
  d_kmer_positions(max_chars_per_batch) {}

auto Searcher::search(
  const vector<u64> &bit_seqs,
  const vector<u64> &kmer_positions,
  vector<u64> &results,
  u64 batch_id
) -> void {
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format("Batch {} consists of {} queries", batch_id, kmer_positions.size())
  );
  copy_to_gpu(batch_id, bit_seqs, kmer_positions, results);
  if (!kmer_positions.empty()) {
    Logger::log_timed_event(
      "SearcherSearch", Logger::EVENT_STATE::START, format("batch {}", batch_id)
    );
    launch_search_kernel(kmer_positions.size(), batch_id);
    Logger::log_timed_event(
      "SearcherSearch", Logger::EVENT_STATE::STOP, format("batch {}", batch_id)
    );
    copy_from_gpu(results, batch_id, kmer_positions);
  }
}

auto Searcher::copy_to_gpu(
  u64 batch_id,
  const vector<u64> &bit_seqs,
  const vector<u64> &kmer_positions,
  vector<u64> &results
) -> void {
  Logger::log_timed_event(
    "SearcherCopyToGpu",
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  d_bit_seqs.set(bit_seqs, bit_seqs.size());
  auto padded_query_size
    = round_up<u64>(kmer_positions.size(), superblock_bits);
  d_kmer_positions.set(kmer_positions.data(), kmer_positions.size());
  d_kmer_positions.memset(
    kmer_positions.size(), padded_query_size - kmer_positions.size(), 0
  );
  Logger::log_timed_event(
    "SearcherCopyToGpu", Logger::EVENT_STATE::STOP, format("batch {}", batch_id)
  );
  results.resize(kmer_positions.size());
}

auto Searcher::copy_from_gpu(
  vector<u64> &results, u64 batch_id, const vector<u64> &kmer_positions
) -> void {
  Logger::log_timed_event(
    "SearcherCopyFromGpu",
    Logger::EVENT_STATE::START,
    format("batch {}", batch_id)
  );
  d_kmer_positions.copy_to(results.data(), kmer_positions.size());
  Logger::log_timed_event(
    "SearcherCopyFromGpu",
    Logger::EVENT_STATE::STOP,
    format("batch {}", batch_id)
  );
}

}  // namespace sbwt_search
