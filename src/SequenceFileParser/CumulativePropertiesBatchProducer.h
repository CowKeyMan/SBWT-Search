#ifndef CUMULATIVE_PROPERTIES_BATCH_PRODUCER_H
#define CUMULATIVE_PROPERTIES_BATCH_PRODUCER_H

/**
 * @file CumulativePropertiesBatchProducer.h
 * @brief Builds the CimulativeProperties Batches
 * */

#include <memory>
#include <string>

#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class CumulativePropertiesBatch;
}  // namespace sbwt_search

using std::make_shared;
using std::shared_ptr;
using std::string;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

class CumulativePropertiesBatchProducer {
  private:
    CircularBuffer<shared_ptr<CumulativePropertiesBatch>> batches;
    bool finished_reading = false;
    BoundedSemaphore semaphore;
    uint kmer_size;

  public:
    CumulativePropertiesBatchProducer(
      u64 max_batches, u64 max_strings_per_batch, uint kmer_size
    );
    void add_string(const string &s);
    void terminate_batch();
    void start_new_batch();
    void set_finished_reading();
    bool operator>>(shared_ptr<CumulativePropertiesBatch> &batch);

  private:
    shared_ptr<CumulativePropertiesBatch>
    get_empty_cumsum_batch(const u64 max_strings_per_batch);
    void reset_batch(shared_ptr<CumulativePropertiesBatch> &batch);
    bool no_more_sequences();
};
}  // namespace sbwt_search

#endif
