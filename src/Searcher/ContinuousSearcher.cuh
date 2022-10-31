#ifndef CONTINUOUS_SEARCHER_CUH
#define CONTINUOUS_SEARCHER_CUH

/**
 * @file ContinuousSearcher.cuh
 * @brief Search implementation with threads
 * */

#include <memory>

#include "BatchObjects/BitSeqBatch.h"
#include "BatchObjects/PositionsBatch.h"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "Searcher/Searcher.cuh"
#include "Utils/GlobalDefinitions.h"
#include "Utils/Logger.h"
#include "Utils/MathUtils.hpp"
#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using design_utils::SharedBatchesProducer;
using fmt::format;
using log_utils::Logger;
using math_utils::round_up;
using std::shared_ptr;

namespace sbwt_search {

template <class PositionsProducer, class BitSeqProducer>
class ContinuousSearcher: public SharedBatchesProducer<vector<u64>> {
    SearcherGpu searcher;
    shared_ptr<BitSeqProducer> bit_seq_producer;
    shared_ptr<PositionsProducer> positions_producer;
    shared_ptr<BitSeqBatch> bit_seq_batch;
    shared_ptr<PositionsBatch> kmer_positions;
    u64 max_positions_per_batch;

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
        max_positions_per_batch(max_positions_per_batch),
        SharedBatchesProducer<vector<u64>>(max_batches) {
      initialise_batches();
    }

    auto get_default_value() -> shared_ptr<vector<u64>> override {
      return make_shared<vector<u64>>(
        round_up<u64>(max_positions_per_batch, superblock_bits)
      );
    }

    auto continue_read_condition() -> bool override {
      return (*positions_producer >> kmer_positions)
           & (*bit_seq_producer >> bit_seq_batch);
    }

    auto generate() -> void override {
      searcher.search(
        bit_seq_batch->bit_seq,
        kmer_positions->positions,
        *batches.current_write(),
        batch_id
      );
    }

    auto do_at_batch_start() -> void override {
      SharedBatchesProducer<vector<u64>>::do_at_batch_start();
      Logger::log_timed_event(
        "Searcher", Logger::EVENT_STATE::START, format("batch {}", batch_id)
      );
    }

    auto do_at_batch_finish() -> void override {
      Logger::log_timed_event(
        "Searcher", Logger::EVENT_STATE::STOP, format("batch {}", batch_id)
      );
      SharedBatchesProducer<vector<u64>>::do_at_batch_finish();
    }
};

}  // namespace sbwt_search
#endif
