#ifndef CUMULATIVE_PROPERTIES_BATCH_PRODUCER_HPP
#define CUMULATIVE_PROPERTIES_BATCH_PRODUCER_HPP

/**
 * @file stringsequencebatchbuilder.hpp
 * @brief takes care of building and sending the stringsequencebatch
 * */

class CumulativePropertiesBatchProducer {
    CircularBuffer<shared_ptr<CumulativePropertiesBatch>> cumsum_batches;
    CumulativePropertiesBatchProducer(): cumsum_batches(max_batches + 1) {
      for (int i = 0; i < max_batches; ++i) {
        /* cumsum_batches.set(i, get_empty_cumsum_batch()); */
      }
    }

    auto get_empty_cumsum_batch() -> shared_ptr<CumulativePropertiesBatch> {
      auto batch = make_shared<CumulativePropertiesBatch>();
      batch->cumsum_positions_per_string.reserve(max_strings_per_batch);
      batch->cumsum_positions_per_string.push_back(0);
      batch->cumsum_string_lengths.reserve(max_strings_per_batch);
      batch->cumsum_string_lengths.push_back(0);
      return batch;
    }
};

#endif
