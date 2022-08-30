#ifndef STRING_SEQUENCE_BATCH_PRODUCER_H
#define STRING_SEQUENCE_BATCH_PRODUCER_H

/**
 * @file StringSequenceBatchProducer.h
 * @brief takes care of building and sending the stringsequencebatch
 * */

#include <memory>
#include <stddef.h>
#include <string>

#include "Utils/SharedBatchesProducer.hpp"
#include "Utils/TypeDefinitions.h"

namespace sbwt_search {
class StringSequenceBatch;
class ContinuousSequenceFileParser;
}  // namespace sbwt_search

using design_utils::SharedBatchesProducer;
using std::shared_ptr;
using std::string;

namespace sbwt_search {

class StringSequenceBatchProducer:
    public SharedBatchesProducer<StringSequenceBatch> {
    friend ContinuousSequenceFileParser;

  private:
    const uint bits_split = 64;
    const uint max_strings_per_batch;
    const uint num_readers;
    uint current_batch_size = 0;
    uint chars_to_next_index;
    const uint chars_per_reader;

    StringSequenceBatchProducer(
      const size_t max_strings_per_batch,
      const size_t max_chars_per_batch,
      const uint max_batches,
      const uint num_readers,
      const uint bits_split = 64
    );
    void add_string(const string &s);
    shared_ptr<StringSequenceBatch> get_default_value() override;
    void reset_batch(shared_ptr<StringSequenceBatch> &batch);
    void do_at_batch_start(unsigned int batch_id = 0) override;
    void do_at_batch_finish(unsigned int batch_id = 0) override;
};

}  // namespace sbwt_search
#endif
