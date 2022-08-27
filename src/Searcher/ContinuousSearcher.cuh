#ifndef CONTINUOUS_SEARCHER_CUH
#define CONTINUOUS_SEARCHER_CUH

/**
 * @file ContinuousSearcher.cuh
 * @brief Search implementation with threads
 * */

#include <memory>

#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "Searcher/Searcher.cuh"
#include "Utils/Logger.h"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using fmt::format;
using log_utils::Logger;
using std::shared_ptr;

namespace sbwt_search {

template <
  class PositionsProducer,
  class BitSeqProducer,
  u32 threads_per_block,
  u64 superblock_bits,
  u64 hyperblock_bits,
  u32 presearch_letters,
  bool reversed_bits>
class ContinuousSearcher {
    SearcherGpu<
      threads_per_block,
      superblock_bits,
      hyperblock_bits,
      presearch_letters,
      reversed_bits>
      searcher;
    shared_ptr<BitSeqProducer> bit_seq_producer;
    shared_ptr<PositionsProducer> positions_producer;
    CircularBuffer<shared_ptr<vector<u64>>> batches;
    bool finished = false;
    BoundedSemaphore semaphore;

  public:
    ContinuousSearcher(
      shared_ptr<GpuSbwtContainer> container,
      shared_ptr<BitSeqProducer> bit_seq_producer,
      shared_ptr<PositionsProducer> positions_producer,
      uint max_batches,
      uint max_positions_per_batch
    ):
        searcher(move(container)),
        bit_seq_producer(bit_seq_producer),
        positions_producer(positions_producer),
        semaphore(0, max_batches),
        batches(max_batches + 1) {
      for (uint i = 0; i < batches.size(); ++i) {
        batches.set(
          i,
          make_shared<vector<u64>>(
            round_up<u64>(max_positions_per_batch, superblock_bits)
          )
        );
      }
    }

    void read_and_generate() {
      shared_ptr<vector<u64>> bit_seqs, kmer_positions;
      for (uint batch_idx = 0; (*bit_seq_producer >> bit_seqs)
                               & (*positions_producer >> kmer_positions);
           ++batch_idx) {
        Logger::log_timed_event(
          "Searcher", Logger::EVENT_STATE::START, format("batch {}", batch_idx)
        );
        searcher.search(*bit_seqs, *kmer_positions, *batches.current_write());
        Logger::log_timed_event(
          "Searcher", Logger::EVENT_STATE::STOP, format("batch {}", batch_idx)
        );
        batches.step_write();
        semaphore.release();
      }
      finished = true;
      semaphore.release();
    }

    bool operator>>(shared_ptr<vector<u64>> &results) {
      semaphore.acquire();
      if (finished && batches.empty()) { return false; }
      results = batches.current_read();
      batches.step_read();
      return true;
    }
};

}  // namespace sbwt_search
#endif
