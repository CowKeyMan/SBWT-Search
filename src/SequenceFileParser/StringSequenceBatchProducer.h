#ifndef STRING_SEQUENCE_BATCH_PRODUCER_H
#define STRING_SEQUENCE_BATCH_PRODUCER_H

/**
 * @file StringSequenceBatchProducer.h
 * @brief Takes care of building and producing the stringsequencebatch. This is
 * handed over as a pointer from the ContinuousSequenceFileParser, to be shared
 * with the next consumer component
 */

#include <memory>
#include <string>

#include "BatchObjects/StringSequenceBatch.h"
#include "Tools/SharedBatchesProducer.hpp"
#include "Tools/TypeDefinitions.h"

namespace sbwt_search {

class ContinuousSequenceFileParser;

using design_utils::SharedBatchesProducer;
using std::shared_ptr;
using std::vector;

class StringSequenceBatchProducer:
    public SharedBatchesProducer<StringSequenceBatch> {
  using Base = SharedBatchesProducer<StringSequenceBatch>;
  friend ContinuousSequenceFileParser;

public:
  explicit StringSequenceBatchProducer(u64 max_batches);

  auto static get_bits_per_element() -> u64;

private:
  void set_string(const vector<char> &s);
  auto get_default_value() -> shared_ptr<StringSequenceBatch> override;
};

}  // namespace sbwt_search
#endif
