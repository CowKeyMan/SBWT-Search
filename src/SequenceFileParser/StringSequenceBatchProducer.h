#ifndef STRING_SEQUENCE_BATCH_PRODUCER_H
#define STRING_SEQUENCE_BATCH_PRODUCER_H

/**
 * @file StringSequenceBatchProducer.h
 * @brief takes care of building and sending the stringsequencebatch
 * */

#include <memory>
#include <stddef.h>
#include <string>

#include "Utils/BoundedSemaphore.hpp"
#include "Utils/CircularBuffer.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class StringSequenceBatch;
}  // namespace sbwt_search

using std::shared_ptr;
using std::string;
using structure_utils::CircularBuffer;
using threading_utils::BoundedSemaphore;

namespace sbwt_search {

class StringSequenceBatchProducer {
  private:
    const uint bits_split = 64;
    CircularBuffer<shared_ptr<StringSequenceBatch>> batches;
    const uint max_strings_per_batch;
    const uint num_readers;
    BoundedSemaphore semaphore;
    uint current_batch_size = 0;
    bool finished_reading = false;
    uint chars_to_next_index;
    const uint chars_per_reader;

  public:
    StringSequenceBatchProducer(
      const size_t max_strings_per_batch,
      const size_t max_chars_per_batch,
      const uint max_batches,
      const uint num_readers,
      const uint bits_split = 64
    );
    void add_string(const string &s);
    void terminate_batch();
    void start_new_batch();
    void set_finished_reading();
    bool operator>>(shared_ptr<StringSequenceBatch> &batch);

  private:
    shared_ptr<StringSequenceBatch> get_empty_sequence_batch();
    void reset_batch(shared_ptr<StringSequenceBatch> &batch);
    bool no_more_sequences();
};

}  // namespace sbwt_search
#endif
